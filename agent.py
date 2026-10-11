import operator
import os
import re
from typing import Annotated, Callable, Literal, TypedDict

from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

TOP_K = int(os.getenv("TOP_K", "4"))  # passages retrieved per search
MAX_SEARCHES = int(os.getenv("MAX_SEARCHES", "2"))  # searches per claim, including the retry
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))  # parallel LLM calls
# How the LLM is made to return structured data: "function_calling", "json_schema",
# or empty for the provider's default.
STRUCTURED_OUTPUT = os.getenv("STRUCTURED_OUTPUT", "")

# --- What the LLM must return at each step (structured output) ---------------


class Claim(BaseModel):
    statement: str = Field(
        description="One self-contained sentence stating the fact, including its subject and the specific value."
    )


class Claims(BaseModel):
    claims: list[Claim]


class Verdict(BaseModel):
    passage: int = Field(description="Number of the passage this verdict is about.")
    relation: Literal["conflict", "agree", "unrelated"]
    quote: str = Field(description="The sentence copied from the passage that the verdict rests on. Empty if unrelated.")
    explanation: str = Field(description="One sentence explaining the verdict.")


class Judgement(BaseModel):
    verdicts: list[Verdict]
    new_query: str = Field(
        default="",
        description="Only if none of the passages are about the claim's topic and a differently "
        "worded search could find one: that search query. Otherwise empty.",
    )


class Verification(BaseModel):
    quote_a: str = Field(description="Text copied exactly from passage A that states its side.")
    quote_b: str = Field(description="Text copied exactly from passage B that states its side.")
    is_conflict: bool
    reason: str = Field(description="One sentence: why this is or is not a real contradiction.")


# --- Prompts -------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You extract checkable claims from a passage of a business document so they can be "
    "compared with other documents.\n"
    "- Extract specific facts: numbers, amounts, dates, deadlines, durations, limits, "
    "rules, and who is responsible for what.\n"
    "- Write each claim as one self-contained sentence that names its subject, so it can "
    "be understood without the passage.\n"
    "- Skip vague or general statements. Extract at most 8 claims. Return an empty list "
    "if there are none."
)

JUDGE_SYSTEM = (
    "You compare a claim from one document with passages from other documents of the "
    "same organisation. For every passage give a verdict:\n"
    "- conflict: the passage is about the same subject and scope as the claim and states "
    "something that cannot be true at the same time.\n"
    "- agree: the passage states the same thing.\n"
    "- unrelated: the passage is about something else, or about a different scope (a "
    "different group of people, product, place or time period).\n"
    "Copy the sentence your verdict rests on into `quote`."
)

VERIFY_SYSTEM = (
    "You are a sceptical reviewer checking a reported contradiction between two passages. "
    "Many reports are false alarms.\n"
    "- Copy the exact text from each passage that states its side into quote_a and quote_b. "
    "Copy it character for character; do not paraphrase.\n"
    "- Set is_conflict to true only if both quotes are about the same subject and the same "
    "scope and cannot both be true. A general rule and an explicitly stated exception or "
    "special case are not a contradiction."
)


def _passage_block(tag: str, passage: dict) -> str:
    return f'<{tag} source="{passage["source"]}" page="{passage["page"]}">\n{passage["text"]}\n</{tag}>'


# --- Graph state -----------------------------------------------------------------
# A "passage" is {"id", "source", "page", "text"}.
# A "finding" is {"claim", "a": passage, "b": passage, "judge_quote", "explanation"};
# verification adds {"quote_a", "quote_b", "verified", "reason"}.


class AuditState(TypedDict, total=False):
    claims: list[dict]
    findings: Annotated[list[dict], operator.add]  # candidates, appended by parallel branches
    reviewed: Annotated[list[dict], operator.add]  # candidates after verification
    conflicts: list[dict]  # verified, deduplicated
    rejected: list[dict]  # failed verification


class InvestigateState(TypedDict, total=False):
    claim: dict  # {"statement", "origin": passage}
    query: str
    searches: int
    seen: list[str]  # ids of passages already judged for this claim
    passages: list[dict]
    findings: Annotated[list[dict], operator.add]


class InvestigateOutput(TypedDict):
    findings: Annotated[list[dict], operator.add]


# --- Helpers ---------------------------------------------------------------------


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .\"'“”‘’").lower()


def quote_in_passage(quote: str, passage_text: str) -> bool:
    """Grounding check: is the quote really in the passage (ignoring whitespace and case)?"""
    quote = _normalise(quote)
    return bool(quote) and quote in _normalise(passage_text)


def _same_conflict(x: dict, y: dict) -> bool:
    """True if x and y report the same conflict, possibly found from opposite sides."""
    sides_x = {x["a"]["id"]: _normalise(x["quote_a"]), x["b"]["id"]: _normalise(x["quote_b"])}
    sides_y = {y["a"]["id"]: _normalise(y["quote_a"]), y["b"]["id"]: _normalise(y["quote_b"])}
    if sides_x.keys() != sides_y.keys():
        return False
    return all(sides_x[k] in sides_y[k] or sides_y[k] in sides_x[k] for k in sides_x)


# --- The graph -------------------------------------------------------------------


def structured(llm, schema):
    """The LLM, set up to return `schema` and to retry a failed call twice.

    Groq models default to "json_schema": with function calling, small models
    sometimes misspell the function name and the request is rejected.
    """
    method = STRUCTURED_OUTPUT or ("json_schema" if type(llm).__name__ == "ChatGroq" else "")
    runnable = llm.with_structured_output(schema, method=method) if method else llm.with_structured_output(schema)
    return runnable.with_retry(stop_after_attempt=3)


def build_graph(llm, store: Chroma):
    extractor = structured(llm, Claims)
    judge_llm = structured(llm, Judgement)
    verifier = structured(llm, Verification)

    # ---- extract ----
    def extract(state: AuditState) -> dict:
        data = store.get(include=["documents", "metadatas"])
        passages = [
            {"id": i, "source": m["source"], "page": m["page"], "text": t}
            for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        prompts = [[("system", EXTRACT_SYSTEM), ("human", _passage_block("passage", p))] for p in passages]
        results = extractor.batch(prompts, config={"max_concurrency": MAX_CONCURRENCY})
        claims = [
            {"statement": claim.statement, "origin": passage}
            for passage, result in zip(passages, results)
            for claim in result.claims
        ]
        return {"claims": claims}

    def fan_out_claims(state: AuditState):
        return [
            Send("investigate", {"claim": c, "query": c["statement"], "searches": 0})
            for c in state["claims"]
        ] or "collect"

    # ---- investigate (sub-graph with a search/judge loop) ----
    def search(state: InvestigateState) -> dict:
        origin = state["claim"]["origin"]
        seen = state.get("seen", [])
        found = store.similarity_search(state["query"], k=TOP_K, filter={"source": {"$ne": origin["source"]}})
        passages = [
            {"id": d.id, "source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content}
            for d in found
            if d.id not in seen
        ]
        return {
            "passages": passages,
            "seen": seen + [p["id"] for p in passages],
            "searches": state["searches"] + 1,
        }

    def judge(state: InvestigateState) -> dict:
        claim, passages = state["claim"], state["passages"]
        if not passages:
            return {"query": ""}
        numbered = "\n\n".join(
            f'<passage number="{n}" source="{p["source"]}" page="{p["page"]}">\n{p["text"]}\n</passage>'
            for n, p in enumerate(passages, start=1)
        )
        human = f'<claim source="{claim["origin"]["source"]}">\n{claim["statement"]}\n</claim>\n\n{numbered}'
        result = judge_llm.invoke([("system", JUDGE_SYSTEM), ("human", human)])
        findings = [
            {
                "claim": claim["statement"],
                "a": claim["origin"],
                "b": passages[v.passage - 1],
                "judge_quote": v.quote,
                "explanation": v.explanation,
            }
            for v in result.verdicts
            if v.relation == "conflict" and 1 <= v.passage <= len(passages)
        ]
        # Only search again if nothing on-topic was found at all.
        on_topic = any(v.relation != "unrelated" for v in result.verdicts)
        return {"findings": findings, "query": "" if on_topic else result.new_query.strip()}

    def after_judge(state: InvestigateState) -> str:
        return "search" if state["query"] and state["searches"] < MAX_SEARCHES else END

    investigate = StateGraph(InvestigateState, output_schema=InvestigateOutput)
    investigate.add_node("search", search)
    investigate.add_node("judge", judge)
    investigate.add_edge(START, "search")
    investigate.add_edge("search", "judge")
    investigate.add_conditional_edges("judge", after_judge, ["search", END])

    # ---- verify ----
    def collect(state: AuditState) -> dict:
        return {}  # join point: all investigate branches have finished

    def fan_out_findings(state: AuditState):
        return [Send("verify", {"finding": f}) for f in state.get("findings", [])] or "report"

    def verify(payload: dict) -> dict:
        finding = payload["finding"]
        human = (
            f'Reported contradiction: {finding["explanation"]}\n\n'
            f'{_passage_block("passage_a", finding["a"])}\n\n{_passage_block("passage_b", finding["b"])}'
        )
        result = verifier.invoke([("system", VERIFY_SYSTEM), ("human", human)])
        verified, reason = result.is_conflict, result.reason
        if verified and not (
            quote_in_passage(result.quote_a, finding["a"]["text"])
            and quote_in_passage(result.quote_b, finding["b"]["text"])
        ):
            verified, reason = False, "Rejected automatically: a quoted text was not found in its passage."
        reviewed = {
            **finding,
            "quote_a": result.quote_a,
            "quote_b": result.quote_b,
            "verified": verified,
            "reason": reason,
        }
        return {"reviewed": [reviewed]}

    # ---- report ----
    def report(state: AuditState) -> dict:
        conflicts: list[dict] = []
        for item in state.get("reviewed", []):
            if item["verified"] and not any(_same_conflict(item, kept) for kept in conflicts):
                conflicts.append(item)
        conflicts.sort(key=lambda c: (c["a"]["source"], c["a"]["page"], c["b"]["source"], c["b"]["page"]))
        rejected = [item for item in state.get("reviewed", []) if not item["verified"]]
        return {"conflicts": conflicts, "rejected": rejected}

    graph = StateGraph(AuditState)
    graph.add_node("extract", extract)
    graph.add_node("investigate", investigate.compile())
    graph.add_node("collect", collect)
    graph.add_node("verify", verify)
    graph.add_node("report", report)
    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", fan_out_claims, ["investigate", "collect"])
    graph.add_edge("investigate", "collect")
    graph.add_conditional_edges("collect", fan_out_findings, ["verify", "report"])
    graph.add_edge("verify", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_audit(llm, store: Chroma, on_step: Callable[[str, dict], None] | None = None) -> dict:
    """Run the audit and return the final state.

    on_step(node_name, node_output) is called each time a node finishes
    (once per claim for "investigate", once per candidate for "verify"),
    which the UI uses to show progress.
    """
    graph = build_graph(llm, store)
    state: dict = {}
    config = {"max_concurrency": MAX_CONCURRENCY}
    for mode, payload in graph.stream({}, config=config, stream_mode=["updates", "values"]):
        if mode == "values":
            state = payload
        elif on_step:
            for node, output in payload.items():
                on_step(node, output or {})
    return state


def report_markdown(conflicts: list[dict]) -> str:
    lines = ["# Document audit report", "", f"{len(conflicts)} conflict(s) found.", ""]
    for number, c in enumerate(conflicts, start=1):
        lines += [
            f"## {number}. {c['a']['source']} (page {c['a']['page']}) vs {c['b']['source']} (page {c['b']['page']})",
            "",
            f"- **{c['a']['source']}, page {c['a']['page']}:** \"{c['quote_a']}\"",
            f"- **{c['b']['source']}, page {c['b']['page']}:** \"{c['quote_b']}\"",
            "",
            c["reason"],
            "",
        ]
    return "\n".join(lines)