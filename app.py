import os
import re
import sqlite3
import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

DB_PATH = Path("hoa_documents.db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "google/flan-t5-small"
MAX_CHUNK_WORDS = 120
CHUNK_OVERLAP = 24
TOP_K = 3

st.set_page_config(page_title="HOA Document AI", layout="wide")


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_generation_pipeline():
    try:
        return pipeline("text2text-generation", model=GENERATION_MODEL_NAME, device=-1)
    except Exception:
        return None


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            num_pages INTEGER NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    conn.commit()
    return conn


def split_text_to_chunks(text, max_words=MAX_CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    normalized_text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized_text)

    chunks = []
    current_words = []

    def push_chunk():
        if current_words:
            chunks.append(" ".join(current_words).strip())

    def flush_current_words(reserve=0):
        if not current_words:
            return
        if reserve > 0 and len(current_words) > reserve:
            keep = current_words[-reserve:]
            chunks.append(" ".join(current_words[:-reserve]).strip())
            del current_words[:-reserve]
        else:
            push_chunk()
            current_words.clear()

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        if len(sentence_words) > max_words:
            idx = 0
            while idx < len(sentence_words):
                remaining = sentence_words[idx: idx + max_words]
                if current_words and len(current_words) + len(remaining) > max_words:
                    flush_current_words(overlap)
                current_words.extend(remaining)
                if len(current_words) >= max_words:
                    flush_current_words(overlap)
                idx += max_words
            continue

        if len(current_words) + len(sentence_words) > max_words and current_words:
            flush_current_words(overlap)

        current_words.extend(sentence_words)

    push_chunk()
    return chunks if chunks else [normalized_text]


def extract_pdf_text(reader):
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_pages.append(page_text)

    text = "\n\n".join(text_pages).strip()
    field_lines = []

    try:
        fields = reader.get_fields()
    except Exception:
        fields = None

    if fields:
        for name, field_data in fields.items():
            value = None
            if isinstance(field_data, dict):
                value = field_data.get("/V") or field_data.get("V") or field_data.get("/DV") or field_data.get("DV")
            elif hasattr(field_data, "get"):
                value = field_data.get("/V") or field_data.get("V")
            if value:
                field_lines.append(f"{name}: {value}")

    if field_lines:
        text = text + "\n\n" + "\n".join(field_lines) if text else "\n".join(field_lines)

    return text


def embed_texts(texts):
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (embeddings / norms).astype(np.float32)


def serialize_embedding(embedding):
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob):
    return np.frombuffer(blob, dtype=np.float32)


def ingest_pdf(conn, uploaded_file):
    reader = PdfReader(uploaded_file)
    text = extract_pdf_text(reader)
    if not text:
        raise ValueError("The uploaded PDF does not contain extractable text.")

    chunks = split_text_to_chunks(text)
    embeddings = embed_texts(chunks)
    uploaded_at = datetime.datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (filename, uploaded_at, num_pages, text) VALUES (?, ?, ?, ?)",
        (uploaded_file.name, uploaded_at, len(reader.pages), text),
    )
    document_id = cur.lastrowid

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        cur.execute(
            "INSERT INTO chunks (document_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
            (document_id, index, chunk, serialize_embedding(embedding)),
        )

    conn.commit()
    return document_id, len(chunks)


def load_documents(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT id, filename, uploaded_at, num_pages FROM documents ORDER BY uploaded_at DESC"
    )
    return cur.fetchall()


def get_document_text(conn, document_id):
    cur = conn.cursor()
    cur.execute("SELECT text FROM documents WHERE id = ?", (document_id,))
    row = cur.fetchone()
    return row[0] if row else ""


def delete_document(conn, document_id):
    cur = conn.cursor()
    cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
    cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()


def count_chunks(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks")
    return cur.fetchone()[0]


def load_chunks(conn, document_ids=None):
    cur = conn.cursor()
    if document_ids is None or len(document_ids) == 0:
        cur.execute(
            "SELECT id, document_id, chunk_index, text, embedding FROM chunks"
        )
    else:
        placeholders = ",".join("?" for _ in document_ids)
        cur.execute(
            f"SELECT id, document_id, chunk_index, text, embedding FROM chunks WHERE document_id IN ({placeholders})",
            document_ids,
        )

    rows = cur.fetchall()
    return [
        {
            "id": row[0],
            "document_id": row[1],
            "chunk_index": row[2],
            "text": row[3],
            "embedding": deserialize_embedding(row[4]),
        }
        for row in rows
    ]


def search_relevant_chunks(query, chunks, top_k=TOP_K):
    if not chunks:
        return []

    query_embedding = embed_texts([query])[0]
    embeddings = np.vstack([chunk["embedding"] for chunk in chunks])
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)

    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), min(top_k, embeddings.shape[0]))
    results = []

    for score, index in zip(distances[0], indices[0]):
        if index < 0:
            continue
        chunk = chunks[index]
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "document_id": chunk["document_id"],
            }
        )

    return results


def extract_date_from_text(text):
    cleaned_text = re.sub(r"_+", " ", text)
    patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{2,4})?\b",
        r"\bthis\s+\d{1,2}\s+day\s+of\s+[A-Za-z]+(?:,?\s*\d{4})?\b",
        r"\bthis\s+day\s+of\s+[A-Za-z]+(?:,?\s*\d{4})?\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if match:
            date_text = match.group(0).strip()
            return re.sub(r"\s+", " ", date_text)
    return None


def extract_date_from_snippets(snippets):
    for snippet in snippets:
        date_value = extract_date_from_text(snippet["text"])
        if date_value:
            return date_value, snippet["text"]
    return None, None


def extract_management_company_from_texts(snippets):
    # Try to find explicit company names in the provided snippets first
    combined = "\n\n".join(s.get("text", "") for s in snippets)

    # Pattern: NAME followed by common entity suffixes (LLC, INC, GROUP, etc.)
    company_pattern = re.compile(r"([A-Z][A-Z0-9 &,\.\-]{4,}?(?:LLC|L\.L\.C\.|INC\.|CORP\.|CO\.|COMPANY|GROUP))", re.MULTILINE)
    m = company_pattern.search(combined)
    if m:
        return m.group(1).strip(), None

    # Pattern: phrase like 'hereafter known as' — capture the name before it
    hereafter_pattern = re.compile(r"([A-Z][A-Z0-9 &,\.\-]{4,}?),?\s+hereafter\s+known\s+as", re.IGNORECASE)
    m2 = hereafter_pattern.search(combined)
    if m2:
        return m2.group(1).strip(), None

    # If not found in snippets, try full documents referenced by snippets
    doc_ids = list({s.get("document_id") for s in snippets if s.get("document_id") is not None})
    if doc_ids:
        cur = sqlite3.connect(DB_PATH, check_same_thread=False).cursor()
        placeholders = ",".join("?" for _ in doc_ids)
        cur.execute(f"SELECT text FROM documents WHERE id IN ({placeholders})", doc_ids)
        rows = cur.fetchall()
        for (full_text,) in rows:
            m = company_pattern.search(full_text)
            if m:
                return m.group(1).strip(), full_text[:400]
            m2 = hereafter_pattern.search(full_text)
            if m2:
                return m2.group(1).strip(), full_text[:400]

    return None, None


def generate_answer(question, snippets):
    question_lower = question.lower()
    generation_pipeline = get_generation_pipeline()
    # Only extract management company names when the user explicitly asks about the manager/company
    if re.search(r"\b(management company|management agent|management firm|manager)\b", question_lower) and re.search(r"\b(who|what|which|name|identify)\b", question_lower):
        company_name, company_context = extract_management_company_from_texts(snippets)
        if company_name:
            if company_context:
                return f"The management company appears to be {company_name} (found in document text: {company_context[:320]}{'...' if len(company_context) > 320 else ''})"
            # find which snippet contained it for citation
            for idx, s in enumerate(snippets, start=1):
                if company_name in s.get("text", ""):
                    return f"The management company appears to be {company_name} (found in Excerpt {idx})."
            return f"The management company appears to be {company_name}."
    prompt = "Answer the question using only the information below. If the answer is not contained in the information, say that it cannot be determined from the documents.\n\n"
    for index, snippet in enumerate(snippets, start=1):
        prompt += f"Excerpt {index}: {snippet['text']}\n\n"
    prompt += f"Question: {question}\nAnswer:"

    if generation_pipeline is not None:
        try:
            result = generation_pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=False,
                truncation=True,
            )
            answer_text = result[0].get("generated_text", "").strip()
            if answer_text and not any(placeholder in answer_text for placeholder in ["_", "____", "blank"]):
                return answer_text
        except Exception:
            pass

        if re.search(r"\b(date|when|start|effective|commence|get going|begin|commencement)\b", question_lower):
            date_value, date_snippet = extract_date_from_snippets(snippets)
            if date_value:
                if re.search(r"\bthis\s+\d{1,2}\s+day\s+of\s+[A-Za-z]+\b", date_snippet, re.IGNORECASE):
                    return f"The management agreement appears to have been made on {date_value}. The document text shows the date phrase: '{date_value}'."
                return f"The contract appears to start on {date_value} based on this excerpt: {date_snippet[:320]}{'...' if len(date_snippet) > 320 else ''}"

            # If no date found in the top snippets, try searching full document text for dates
            # Load full documents for the selected snippets' document_ids
            doc_ids = list({s.get("document_id") for s in snippets if s.get("document_id") is not None})
            if doc_ids:
                cur = sqlite3.connect(DB_PATH, check_same_thread=False).cursor()
                placeholders = ",".join("?" for _ in doc_ids)
                cur.execute(f"SELECT text FROM documents WHERE id IN ({placeholders})", doc_ids)
                rows = cur.fetchall()
                for (full_text,) in rows:
                    dv = extract_date_from_text(full_text)
                    if dv:
                        return f"The contract appears to start on {dv} (found in the document body)."

    fallback = "I could not generate an answer automatically. Here are the most relevant excerpts from the documents:\n\n"
    fallback += "\n\n".join(
        f"Excerpt {index + 1}: {snippet['text'][:400]}{'...' if len(snippet['text']) > 400 else ''}"
        for index, snippet in enumerate(snippets)
    )
    return fallback


def render_document_summary(documents, conn):
    if not documents:
        st.sidebar.info("No documents have been ingested yet.")
        return

    for doc in documents:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.markdown(
                f"- **{doc[1]}** — pages: {doc[3]}, uploaded: {doc[2]}"
            )
        with col2:
            if st.button("🗑️", key=f"delete_{doc[0]}", help="Delete this document"):
                delete_document(conn, doc[0])
                st.rerun()


def main():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_context" not in st.session_state:
        st.session_state.last_context = []

    st.title("HOA Document AI Assistant")
    conn = init_db()
    documents = load_documents(conn)

    st.sidebar.header("Database")
    st.sidebar.markdown(f"**Stored PDFs:** {len(documents)}")
    st.sidebar.markdown(f"**Stored chunks:** {count_chunks(conn)}")
    st.sidebar.markdown("---")
    render_document_summary(documents, conn)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Use the upload panel to ingest new HOA PDF files. Ask a question and get a context-aware answer from the stored documents."
    )

    with st.expander("📊 Database Status & Debug"):
        total_chunks = count_chunks(conn)
        st.write(f"Total chunks in database: {total_chunks}")
        if total_chunks == 0:
            st.warning("⚠️ No chunks found! Make sure you've ingested a PDF and it contains extractable text.")
        else:
            st.success(f"✅ Database has {total_chunks} chunks ready for search.")

    with st.expander("Upload and ingest a PDF file"):
        uploaded_file = st.file_uploader("Upload HOA PDF", type=["pdf"])
        if uploaded_file is not None:
            if st.button("Ingest document"):
                try:
                    document_id, chunk_count = ingest_pdf(conn, uploaded_file)
                    st.success(
                        f"Saved '{uploaded_file.name}' with {chunk_count} chunks to the database."
                    )
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

    available_choices = ["All documents"] + [f"{doc[1]} (id={doc[0]})" for doc in documents]
    selected_choice = st.selectbox("Search scope", available_choices)

    selected_doc_ids = None
    if selected_choice != "All documents":
        match = re.search(r"id=(\d+)", selected_choice)
        if match:
            selected_doc_ids = [int(match.group(1))]

    # If a specific document is selected, allow viewing its extracted text to debug blanks
    if selected_doc_ids and len(selected_doc_ids) == 1:
        doc_id = selected_doc_ids[0]
        with st.expander("View extracted text for selected document"):
            full_text = get_document_text(conn, doc_id)
            if not full_text:
                st.info("No extracted text is stored for this document.")
            else:
                st.text_area("Extracted document text (first 20k chars)", value=full_text[:20000], height=300)
                if "_" in full_text or "___" in full_text:
                    st.warning("The extracted text contains underscores or blanks. If the date was handwritten or part of a visual overlay, consider using OCR (pytesseract) to extract visible text from the PDF pages.")
                    st.info("I can add an OCR fallback that rasterizes pages and runs Tesseract, but it requires Tesseract installed on the host and the Python packages 'pillow' and 'pytesseract'. Would you like me to add that?")

    with st.form(key="query_form", clear_on_submit=True):
        query = st.text_input("Ask a question about your HOA documents", key="question_input")
        submitted = st.form_submit_button("Search")
        if submitted:
            if not query:
                st.warning("Please enter a question before searching.")
            else:
                # Ensure repeated queries trigger a fresh retrieval
                st.session_state.last_answer = ""
                st.session_state.last_context = []
                with st.spinner("Retrieving relevant document passages..."):
                    chunks = load_chunks(conn, selected_doc_ids)
                    # debug: store number of available chunks
                    st.session_state._debug_chunk_count = len(chunks)
                    top_chunks = search_relevant_chunks(query, chunks)

                if not top_chunks:
                    st.warning("No searchable document text was found. Please ingest a PDF first.")
                    st.session_state.last_answer = ""
                    st.session_state.last_query = query
                    st.session_state.last_context = []
                else:
                    answer = generate_answer(query, top_chunks)
                    st.session_state.last_answer = answer
                    st.session_state.last_query = query
                    st.session_state.last_context = [chunk["text"] for chunk in top_chunks]
                    st.session_state.history.append({"question": query, "answer": answer})
                        # clear the input explicitly to avoid stale comparisons elsewhere

    # Show debug info after form (outside form scope)
    if st.session_state.get("last_query"):
        with st.expander(f"🔍 Retrieved chunks for: '{st.session_state.last_query}'", expanded=False):
            if st.session_state.last_context:
                for idx, chunk_text in enumerate(st.session_state.last_context, start=1):
                    st.write(f"**Chunk {idx}:**")
                    st.write(chunk_text[:500])
                    st.divider()
            else:
                st.info("No chunks were retrieved.")

    if st.session_state.last_answer:
        st.markdown("---")
        st.header("Answer")
        if st.session_state.get("last_query"):
            st.subheader(f"Question: {st.session_state.last_query}")
        st.write(st.session_state.last_answer)

        if st.session_state.last_context:
            st.subheader("Retrieved document excerpts")
            for idx, context_text in enumerate(st.session_state.last_context, start=1):
                st.markdown(f"**Excerpt {idx}:**")
                st.write(context_text)

    if st.session_state.get("history"):
        st.markdown("---")
        st.header("Conversation History")
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}\n")

    if st.button("Clear conversation history"):
        st.session_state.history = []
        st.session_state.last_answer = ""
        st.session_state.last_query = ""
        st.session_state.last_context = []
        st.rerun()


if __name__ == "__main__":
    main()
