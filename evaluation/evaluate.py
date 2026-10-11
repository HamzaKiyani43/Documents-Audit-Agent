"""Measure the agent on the test documents with planted contradictions.

Run: python evaluation/evaluate.py            (real models; needs an API key)
     python evaluation/evaluate.py --offline  (rule-based stand-ins; checks the
                                               plumbing only, not model quality)
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(HERE.parent / ".env")

import agent  # noqa: E402
import indexing  # noqa: E402


def matched_id(item: dict, ground_truth: list[dict]) -> str | None:
    """The planted contradiction this reported conflict corresponds to, if any.

    It must be between the same two documents and mention both planted values.
    """
    sources = {item["a"]["source"], item["b"]["source"]}
    text = " ".join(
        item.get(key, "") for key in ("claim", "judge_quote", "explanation", "quote_a", "quote_b", "reason")
    ).lower()
    for truth in ground_truth:
        if sources != {truth["a"]["source"], truth["b"]["source"]}:
            continue
        if all(any(value in text for value in side["values"]) for side in (truth["a"], truth["b"])):
            return truth["id"]
    return None


def score(items: list[dict], ground_truth: list[dict]) -> dict:
    """Count distinct reports: the same planted contradiction (or the same false
    alarm between the same two passages) is counted once, however often reported."""
    reports = {(frozenset((i["a"]["id"], i["b"]["id"])), matched_id(i, ground_truth)) for i in items}
    found = {truth_id for _, truth_id in reports if truth_id}
    false_positives = sum(1 for _, truth_id in reports if truth_id is None)
    return {
        "found": len(found),
        "planted": len(ground_truth),
        "false_positives": false_positives,
        "recall": len(found) / len(ground_truth),
        "precision": len(found) / (len(found) + false_positives) if found or false_positives else 0.0,
        "missed": sorted({t["id"] for t in ground_truth} - found),
    }


def progress_printer():
    """Returns a function that prints a progress line each time a step of the agent finishes."""
    counts = {"claims": 0, "checked": 0, "candidates": 0, "verified": 0}

    def on_step(node: str, output: dict) -> None:
        if node == "extract":
            counts["claims"] = len(output.get("claims", []))
            print(f"Extracted {counts['claims']} claims. Checking each against the other documents...")
        elif node == "investigate":
            counts["checked"] += 1
            counts["candidates"] += len(output.get("findings", []))
            print(f"\r  claim {counts['checked']} of {counts['claims']}", end="", flush=True)
        elif node == "collect":
            print(f"\n{counts['candidates']} candidate conflicts. Verifying each...")
        elif node == "verify":
            counts["verified"] += 1
            print(f"\r  candidate {counts['verified']} of {counts['candidates']}", end="", flush=True)
        elif node == "report":
            print("\nDone.\n")

    return on_step


def evaluate(llm, embeddings) -> dict:
    docs_dir = HERE / "docs"
    if not docs_dir.exists():
        import make_test_docs

        make_test_docs.main()
    ground_truth = json.loads((HERE / "ground_truth.json").read_text())

    with tempfile.TemporaryDirectory() as tmp:
        store = indexing.get_vector_store(embeddings, tmp)
        passages = 0
        for pdf in sorted(docs_dir.glob("*.pdf")):
            passages += indexing.index_pdf(store, pdf.read_bytes(), pdf.name)
        print(f"Indexed {passages} passages. Extracting claims (one LLM call per passage)...", flush=True)
        state = agent.run_audit(llm, store, on_step=progress_printer())

    conflicts = state.get("conflicts", [])
    return {
        "claims_extracted": len(state.get("claims", [])),
        "before_verification": score(state.get("reviewed", []), ground_truth),
        "after_verification": score(conflicts, ground_truth),
        "false_positive_details": [
            {"a": f"{c['a']['source']} p.{c['a']['page']}", "quote_a": c["quote_a"],
             "b": f"{c['b']['source']} p.{c['b']['page']}", "quote_b": c["quote_b"]}
            for c in conflicts
            if matched_id(c, ground_truth) is None
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="use rule-based stand-ins instead of real models")
    args = parser.parse_args()

    if args.offline:
        sys.path.insert(0, str(HERE.parent / "tests"))
        from fakes import RuleBasedLLM, WordHashEmbeddings

        llm, embeddings = RuleBasedLLM(), WordHashEmbeddings()
        print("OFFLINE MODE: rule-based stand-ins, not real models. These numbers only show the pipeline runs.\n")
    else:
        llm, embeddings = indexing.get_llm(), indexing.get_embeddings()
        print(f"Model: {indexing.CHAT_MODEL}   Embeddings: {indexing.EMBEDDING_MODEL}\n")

    results = evaluate(llm, embeddings)

    print(f"Claims extracted: {results['claims_extracted']}\n")
    print(f"{'':22}{'found':>8}{'false +':>9}{'recall':>8}{'precision':>11}")
    for label, key in (("Before verification", "before_verification"), ("After verification", "after_verification")):
        s = results[key]
        print(f"{label:22}{s['found']:>5}/{s['planted']:<2}{s['false_positives']:>9}{s['recall']:>8.0%}{s['precision']:>11.0%}")
    final = results["after_verification"]
    if final["missed"]:
        print("\nMissed:", ", ".join(final["missed"]))
    for fp in results["false_positive_details"]:
        print(f"\nFalse positive: {fp['a']}: \"{fp['quote_a']}\"\n            vs  {fp['b']}: \"{fp['quote_b']}\"")

    name = "results_offline.json" if args.offline else "results.json"
    (HERE / name).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to evaluation/{name}")


if __name__ == "__main__":
    main()