# HOA AI Assistant

A Streamlit application for uploading HOA PDF documents, storing them in a SQLite database, and asking questions via retrieval-augmented chat.

## Features
- Upload PDF files and ingest text into a local database
- Store document text chunks and embeddings for fast retrieval
- Search across all stored HOA documents or a specific file
- Generate answers based on extracted document context

## Setup
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Usage
1. Upload a HOA PDF using the sidebar.
2. Click "Ingest document" to store the document content and embeddings.
3. Ask a question in the query box and click "Search".
4. Review the generated answer and the relevant document excerpts.

## Notes
- The app uses `all-MiniLM-L6-v2` for embedding generation.
- `google/flan-t5-small` is used for local answer generation when available.
- All documents are stored locally in `hoa_documents.db`.
 
