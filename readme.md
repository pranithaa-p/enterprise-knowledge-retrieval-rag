# Enterprise Knowledge Retrieval Portal Using Hybrid Vector-Semantic Search

An enterprise-oriented document retrieval system that combines **BM25 keyword search** and **semantic vector search** to retrieve relevant information from organizational documents. It uses **Reciprocal Rank Fusion (RRF)** to combine retrieval results and a Large Language Model (LLM) to generate contextual answers from retrieved document passages.

## Overview

Organizations maintain large volumes of documents, making it difficult to locate relevant information efficiently. Traditional keyword-based search may overlook relevant content when users express the same concept using different terminology.

This project addresses this challenge through a hybrid retrieval approach that combines lexical matching with semantic similarity, providing ranked document passages and contextual answers through a web-based interface.

## Key Features

- **Document Processing:** Upload and extract text from PDF and UTF-8 TXT documents.
- **Text Preprocessing:** Clean, preprocess, and divide extracted text into manageable chunks.
- **Keyword Search:** Retrieve relevant passages using the BM25 algorithm.
- **Semantic Search:** Generate document embeddings using Sentence Transformers.
- **Vector Similarity Search:** Perform semantic retrieval using FAISS.
- **Hybrid Retrieval:** Combine keyword and semantic search results using Reciprocal Rank Fusion (RRF).
- **Contextual Answer Generation:** Generate natural-language responses using an LLM through the Groq API.
- **Formatted Answers:** Render AI-generated Markdown-style headings, lists, emphasis, links, and code blocks.
- **Source References:** Display the retrieved document excerpts, filenames, chunk numbers, and retrieval scores.

## System Architecture

The system follows a Retrieval-Augmented Generation (RAG) approach.

```text
          Document Upload
                 |
                 v
       Text Extraction & Cleaning
                 |
                 v
           Text Chunking
                 |
          +------+------+
          |             |
          v             v
     BM25 Index     Embeddings
          |             |
          |          FAISS
          |             |
          v             v
      Keyword       Semantic
      Retrieval     Retrieval
          |             |
          +------+------+
                 |
                 v
        Reciprocal Rank Fusion
                 |
                 v
       Ranked Relevant Passages
                 |
                 v
         Context Preparation
                 |
                 v
          Groq LLM API
                 |
                 v
        Contextual Answer
                 |
                 v
          React Frontend
```

## Technology Stack

### Frontend

| Technology | Purpose |
|---|---|
| React 19 | User interface development |
| TypeScript 6 | Type-safe frontend development |
| Vite 8 | Development server and production bundling |
| CSS | Interface styling and responsive layouts |
| Fetch API | Communication with backend APIs |
| Oxlint | Frontend linting |

### Backend

| Technology | Purpose |
|---|---|
| Python 3.12 | Backend programming |
| FastAPI | REST API development |
| Uvicorn | Application server |
| SQLite | Document and metadata storage |
| pypdf | PDF text extraction |
| rank-bm25 | Keyword-based retrieval |
| Sentence Transformers | Semantic embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| NumPy | Numerical and embedding operations |
| Groq Python SDK | LLM API integration |
| openai/gpt-oss-120b | Contextual answer generation |
| python-dotenv | Environment variable loading |
| Pydantic | Request validation |

## Retrieval Methodology

### 1. Keyword-Based Retrieval

BM25 (Best Matching 25) identifies relevant document passages based on lexical matching between the user's query and indexed document content.

### 2. Semantic Retrieval

Sentence Transformers converts document chunks and user queries into dense vector representations using the `all-MiniLM-L6-v2` model.

FAISS performs vector similarity search to identify semantically related passages.

### 3. Hybrid Ranking

The results from BM25 and semantic retrieval are combined using Reciprocal Rank Fusion (RRF).

This approach brings together exact-term matching and semantic relevance in a unified ranking pipeline.

### 4. Answer Generation

The retrieved passages are supplied as context to the Groq-hosted LLM, which generates a natural-language response based on the retrieved information.

## Getting Started

### Prerequisites

- Python 3.12
- Node.js and npm
- Git
- Groq API key

The backend currently accepts `.pdf` and `.txt` files. PDF files are processed with `pypdf`; scanned PDFs without an extractable text layer require OCR before upload.

### Clone the Repository

```bash
git clone https://github.com/pranithaa-p/enterprise-knowledge-retrieval-rag.git

cd enterprise-knowledge-retrieval-rag
```

### Backend Setup

```bash
cd backend

python -m venv .venv
```

Activate the virtual environment:

**Windows:**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file inside the backend directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Start the backend server:

```bash
uvicorn main:app --reload
```

The backend is available at `http://127.0.0.1:8000`. Useful endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Confirm that the API is running |
| GET | `/health` | Check API status and Groq configuration |
| POST | `/documents/extract` | Extract text from one uploaded file |
| POST | `/documents/process` | Process and store one document |
| POST | `/documents/batch-process` | Process multiple documents |
| GET | `/documents` | List stored documents |
| GET | `/documents/{document_id}/chunks` | Retrieve document chunks |
| POST | `/search/hybrid` | Run hybrid retrieval and generate an AI answer |

### Frontend Setup

Open another terminal:

```bash
cd frontend

npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed by Vite in your browser.

The Vite development server proxies `/api` requests to `http://127.0.0.1:8000`, so start the backend before searching or uploading documents.

### Production deployment

For production, do not use `--reload`. Run the backend with a production host and port, for example:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Configure `GROQ_API_KEY` as a backend deployment secret. Never commit `.env` or expose the key in frontend code.

The Vite development proxy is for local development only. When frontend and backend are deployed separately, configure a production API URL or reverse proxy and allow the deployed frontend origin through backend CORS configuration.

The default SQLite database is stored locally in `backend/documents.db`. Use persistent storage or a managed database in production if uploaded documents must survive restarts or redeployments.

## Project Structure

```text
enterprise-knowledge-retrieval-rag/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
│
├── backend/
│   ├── main.py
│   ├── semantic_search.py
│   ├── requirements.txt
│   └── .env  # local only; do not commit
│
├── .gitignore
└── readme.md
```

## Future Enhancements

- Role-Based Access Control (RBAC).
- Company-wise document and data isolation.
- Desktop application packaging.
- Expanded document format support, including DOCX.
- Formal retrieval evaluation using Precision@k, Recall@k, and Mean Reciprocal Rank (MRR).

## Contributors

**Pranitha P** – Backend Development, Document Processing, Hybrid Retrieval, Semantic Search and LLM Integration.

**Nethra B** – Frontend Development, User Interface Design, API Integration and Result Presentation.

## Project Information

- **Project:** S5 Project Work I
- **Academic Year:** 2026–2027
- **Category:** Internal
- **Domain:** Information Retrieval, NLP and Generative AI

---

*Developed as part of the Semester 5 Mini Project, Department of Information Technology.*