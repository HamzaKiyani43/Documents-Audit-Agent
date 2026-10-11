"""Offline tests of the agent graph. Run: pytest"""

import io
import sys
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

import agent  # noqa: E402
import indexing  # noqa: E402
from agent import Claim, Claims, Judgement, Verdict, Verification  # noqa: E402
from fakes import RuleBasedLLM, ScriptedLLM, WordHashEmbeddings, tag_contents  # noqa: E402


def make_pdf(pages: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for text in pages:
        pdf.drawString(50, 800, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


HANDBOOK = make_pdf(["Employees receive 28 days of paid vacation per year."])
GUIDE = make_pdf(["Employees receive 30 days of paid vacation per year.", "The office opens at 8:00 every day."])


@pytest.fixture
def store(tmp_path):
    return indexing.get_vector_store(WordHashEmbeddings(), str(tmp_path / "db"))


@pytest.fixture
def two_docs(store):
    indexing.index_pdf(store, HANDBOOK, "handbook.pdf")
    indexing.index_pdf(store, GUIDE, "guide.pdf")
    return store


def test_finds_conflict_once_with_both_sides_cited(two_docs):
    steps = []
    state = agent.run_audit(RuleBasedLLM(), two_docs, on_step=lambda node, _: steps.append(node))
    # Found from both sides (handbook claim vs guide, guide claim vs handbook) ...
    assert len(state["reviewed"]) == 2
    # ... but reported once.
    [conflict] = state["conflicts"]
    assert {conflict["a"]["source"], conflict["b"]["source"]} == {"handbook.pdf", "guide.pdf"}
    assert {conflict["a"]["page"], conflict["b"]["page"]} == {1}
    assert "28 days" in conflict["quote_a"] + conflict["quote_b"]
    assert "30 days" in conflict["quote_a"] + conflict["quote_b"]
    assert steps[0] == "extract" and steps[-1] == "report" and "investigate" in steps and "verify" in steps


def test_never_compares_a_document_with_itself(store):
    both = make_pdf(["Employees receive 28 days of paid vacation per year.",
                     "Employees receive 30 days of paid vacation per year."])
    indexing.index_pdf(store, both, "only.pdf")
    state = agent.run_audit(RuleBasedLLM(), store)
    assert len(state["claims"]) == 2
    assert state["conflicts"] == [] and state.get("findings", []) == []


def test_empty_index(store):
    state = agent.run_audit(RuleBasedLLM(), store)
    assert state["claims"] == [] and state["conflicts"] == []


def test_searches_again_when_judge_asks_for_new_query(two_docs):
    queries = []
    real_search = two_docs.similarity_search

    def spy(query, **kwargs):
        queries.append(query)
        return real_search(query, **kwargs)

    two_docs.similarity_search = spy

    def extract(prompt):
        is_handbook = 'source="handbook.pdf"' in prompt
        return Claims(claims=[Claim(statement="Annual leave is 28 days.")] if is_handbook else [])

    def judge(prompt):
        passages = tag_contents(prompt, "passage")
        return Judgement(
            verdicts=[Verdict(passage=n, relation="unrelated", quote="", explanation="") for n in range(1, len(passages) + 1)],
            new_query="paid vacation days per year",
        )

    state = agent.run_audit(ScriptedLLM(extract, judge, verify=None), two_docs)
    # One retry, then it stops even though the judge asks again (MAX_SEARCHES = 2).
    assert queries == ["Annual leave is 28 days.", "paid vacation days per year"]
    assert state["conflicts"] == []


def test_invented_quote_is_rejected(two_docs):
    def verify(prompt):
        return Verification(quote_a="Employees receive 28 days", quote_b="Staff get 45 days off",
                            is_conflict=True, reason="They differ.")

    llm = RuleBasedLLM()
    scripted = ScriptedLLM(llm._extract, llm._judge, verify)
    state = agent.run_audit(scripted, two_docs)
    assert state["conflicts"] == []
    assert all("not found in its passage" in r["reason"] for r in state["rejected"])
    assert len(state["rejected"]) == 2


def test_quote_check_ignores_line_breaks_and_case():
    assert agent.quote_in_passage('"Employees receive\n28 days."', "All employees  receive 28 days. More text.")
    assert not agent.quote_in_passage("", "anything")
    assert not agent.quote_in_passage("29 days", "Employees receive 28 days.")


def test_verifier_can_reject(two_docs):
    def verify(prompt):
        return Verification(quote_a="x", quote_b="y", is_conflict=False, reason="Different scope.")

    llm = RuleBasedLLM()
    state = agent.run_audit(ScriptedLLM(llm._extract, llm._judge, verify), two_docs)
    assert state["conflicts"] == [] and len(state["rejected"]) == 2


def test_report_markdown(two_docs):
    state = agent.run_audit(RuleBasedLLM(), two_docs)
    report = agent.report_markdown(state["conflicts"])
    assert "1 conflict(s) found" in report
    assert "handbook.pdf" in report and "guide.pdf" in report and "28 days" in report


def test_evaluation_scoring():
    import evaluate

    truth = [{"id": "vacation", "a": {"source": "h.pdf", "values": ["28"]}, "b": {"source": "g.pdf", "values": ["30"]}}]

    def item(a_src, b_src, a_id, b_id, text):
        return {"a": {"source": a_src, "id": a_id}, "b": {"source": b_src, "id": b_id}, "claim": text}

    hit = item("h.pdf", "g.pdf", "h1", "g1", "28 days vs 30 days")
    mirrored = item("g.pdf", "h.pdf", "g1", "h1", "30 days vs 28 days")
    wrong_docs = item("h.pdf", "x.pdf", "h1", "x1", "28 days vs 30 days")
    result = evaluate.score([hit, mirrored, wrong_docs], truth)
    assert (result["found"], result["false_positives"], result["missed"]) == (1, 1, [])
    assert result["precision"] == 0.5 and result["recall"] == 1.0
    assert evaluate.score([], truth)["missed"] == ["vacation"]
