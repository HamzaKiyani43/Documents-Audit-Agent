"""Streamlit UI for the Document Audit Agent.  Run: streamlit run app.py"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import agent  # noqa: E402  (after load_dotenv so settings from .env apply)
import indexing  # noqa: E402

st.set_page_config(page_title="Document Audit Agent", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def get_store():
    return indexing.get_vector_store(indexing.get_embeddings())


@st.cache_resource
def get_llm():
    return indexing.get_llm()


def side(column, passage, quote):
    column.markdown(f"**{passage['source']}** · page {passage['page']}")
    column.info(quote)
    with column.expander("Full passage"):
        st.caption(passage["text"])


store = get_store()
documents = indexing.list_documents(store)

# --- Sidebar: manage documents ----------------------------------------------

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    if uploads and st.button("Index uploaded files", type="primary"):
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name}…"):
                try:
                    indexing.index_pdf(store, upload.getvalue(), upload.name)
                except Exception as error:
                    st.error(str(error))
        st.session_state.pop("audit", None)  # results no longer match the index
        st.rerun()

    if documents:
        st.subheader("Indexed")
        for name, count in documents.items():
            left, right = st.columns([5, 1])
            left.write(f"{name}  \n:gray[{count} passages]")
            if right.button("", icon=":material/delete:", key=f"delete-{name}", help=f"Remove {name}"):
                indexing.delete_document(store, name)
                st.session_state.pop("audit", None)
                st.rerun()

# --- Run the audit -------------------------------------------------------------

st.title("🔎 Document Audit Agent")
st.caption("Finds statements that contradict each other across your documents. You decide which ones are real.")

if len(documents) < 2:
    st.info("Upload and index at least two PDFs in the sidebar. The agent compares documents with each other.")

if st.button("Run audit", type="primary", disabled=len(documents) < 2):
    bar = st.progress(0.0, text="Reading the documents and extracting claims…")
    progress = {"claims": 0, "checked": 0, "candidates": 0, "verified": 0}

    def on_step(node: str, output: dict) -> None:
        if node == "extract":
            progress["claims"] = len(output.get("claims", []))
        elif node == "investigate":
            progress["checked"] += 1
            progress["candidates"] += len(output.get("findings", []))
            bar.progress(
                0.1 + 0.6 * progress["checked"] / max(progress["claims"], 1),
                text=f"Checking claim {progress['checked']} of {progress['claims']} against the other documents…",
            )
        elif node == "verify":
            progress["verified"] += 1
            bar.progress(
                0.7 + 0.3 * progress["verified"] / max(progress["candidates"], 1),
                text=f"Verifying candidate {progress['verified']} of {progress['candidates']}…",
            )

    try:
        st.session_state.audit = agent.run_audit(get_llm(), store, on_step)
    except Exception as error:
        st.error(f"The audit failed ({type(error).__name__}): {error}")
        st.caption(f"Model: {indexing.CHAT_MODEL}. Check the API key in .env, then stop and restart the app.")
        st.stop()
    for key in [k for k in st.session_state if k.startswith("decision-")]:
        del st.session_state[key]
    bar.empty()

# --- Review the results ----------------------------------------------------------

audit = st.session_state.get("audit")
if audit:
    conflicts, rejected = audit.get("conflicts", []), audit.get("rejected", [])
    one, two, three = st.columns(3)
    one.metric("Claims checked", len(audit.get("claims", [])))
    two.metric("Candidate conflicts", len(audit.get("reviewed", [])))
    three.metric("Conflicts after verification", len(conflicts))

    if not conflicts:
        st.success("No contradictions found between the documents.")

    for number, conflict in enumerate(conflicts, start=1):
        with st.container(border=True):
            st.subheader(f"Conflict {number}")
            left, right = st.columns(2)
            side(left, conflict["a"], conflict["quote_a"])
            side(right, conflict["b"], conflict["quote_b"])
            st.write(conflict["reason"])
            st.segmented_control(
                "Your decision", ["Confirm", "Dismiss"], key=f"decision-{number}", label_visibility="collapsed"
            )

    if conflicts:
        decisions = [st.session_state.get(f"decision-{n}") for n in range(1, len(conflicts) + 1)]
        confirmed = [c for c, d in zip(conflicts, decisions) if d == "Confirm"]
        undecided = decisions.count(None)
        st.divider()
        st.write(f"{len(confirmed)} confirmed, {decisions.count('Dismiss')} dismissed, {undecided} undecided.")
        st.download_button(
            "Download report of confirmed conflicts",
            agent.report_markdown(confirmed),
            file_name="audit_report.md",
            disabled=not confirmed,
        )

    if rejected:
        with st.expander(f"Rejected by the verification step ({len(rejected)})"):
            for item in rejected:
                st.markdown(
                    f"**{item['a']['source']}** p.{item['a']['page']} vs "
                    f"**{item['b']['source']}** p.{item['b']['page']}  \n"
                    f"Claim: {item['claim']}  \n:gray[{item['reason']}]"
                )