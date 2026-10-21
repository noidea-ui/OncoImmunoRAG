# OncoImmunoRAG

A Retrieval-Augmented Generation (RAG) system for **oncology immunotherapy** literature.

## Features
- **Ingestion**: load & clean PDF papers from `data/raw/`.
- **Retrieval**: vector search over `data/vector_db/` (ChromaDB).
- **Grading**: evidence-level scoring of retrieved passages.
- **Evaluation**: harness for retrieval & answer quality.

## Project Layout
```
OncoImmunoRAG/
├── data/
│   ├── raw/           # source PDFs
│   └── vector_db/     # persistent Chroma store
├── src/
│   ├── ingestion/     # load + clean PDFs
│   ├── retrieval/     # vector search
│   ├── grading/       # evidence grading
│   └── evaluation/    # evaluation scripts
├── tests/
└── pyproject.toml
```

## Setup
```bash
poetry install
```

## Usage
```bash
poetry run oncoimmuno-rag
```
