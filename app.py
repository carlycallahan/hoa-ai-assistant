import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap

st.title("HOA Document AI")

uploaded_file = st.file_uploader("Upload HOA PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
         text += page_text

    chunks = textwrap.wrap(text, 500, break_long_words=False)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    query = st.text_input("Ask a question")

    if query:
        q_embed = model.encode([query])
        D, I = index.search(np.array(q_embed), k=3)

        results = [chunks[i] for i in I[0]]

        st.write("Relevant info:")
        for r in results:
            st.write(r)