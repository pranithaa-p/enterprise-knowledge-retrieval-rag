import hashlib
import os
import re
import sqlite3
import traceback
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from groq import Groq

from semantic_search import SemanticSearch


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(
    title="Enterprise Knowledge Retrieval Portal",
    description="Hybrid keyword and semantic document search",
    version="1.3.0",
)

DB_PATH = Path(os.getenv("DATA_DIR", ".")) / "documents.db"

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_BATCH_FILES = 20
MAX_BATCH_SIZE = 100 * 1024 * 1024


# ============================================================
# CUSTOM SWAGGER UPLOAD SCHEMA
# ============================================================

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        openapi_version="3.0.2",
    )

    upload_paths = [
        "/documents/extract",
        "/documents/process",
        "/documents/batch-process",
    ]

    for path in upload_paths:
        if path not in schema.get("paths", {}):
            continue

        post_operation = schema["paths"][path].get("post", {})
        request_body = (
            post_operation.get("requestBody", {})
            .get("content", {})
            .get("multipart/form-data", {})
            .get("schema")
        )

        if not request_body:
            continue

        if "$ref" in request_body:
            schema_name = request_body["$ref"].split("/")[-1]
            request_body = schema["components"]["schemas"].get(
                schema_name, {}
            )

        properties = request_body.setdefault("properties", {})

        if path == "/documents/batch-process":
            properties["files"] = {
                "type": "array",
                "items": {
                    "type": "string",
                    "format": "binary",
                },
            }
        else:
            properties["file"] = {
                "type": "string",
                "format": "binary",
            }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi


# ============================================================
# DATABASE SETUP
# ============================================================

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                page_count INTEGER,
                character_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                content_hash TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            )
            """
        )

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(documents)"
            ).fetchall()
        }

        if "content_hash" not in columns:
            conn.execute(
                "ALTER TABLE documents ADD COLUMN content_hash TEXT"
            )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id)
            """
        )


initialize_database()


# ============================================================
# REQUEST MODEL
# ============================================================

class KeywordSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(filename: str, content: bytes):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        try:
            reader = PdfReader(BytesIO(content))

            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                except Exception:
                    decrypted = 0

                if not decrypted:
                    raise ValueError(
                        "This PDF requires a password to extract text."
                    )

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            return text, len(reader.pages)

        except ValueError:
            raise

        except Exception as exc:
            raise ValueError(
                "Could not read this PDF. It may be damaged or unsupported."
            ) from exc

    if extension == ".txt":
        return content.decode("utf-8-sig"), None

    raise ValueError("Only PDF and TXT files are supported.")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove repeated spaces and tabs without removing newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Prefer a paragraph or sentence boundary near the target.
            min_boundary = start + chunk_size // 2
            candidates = [
                text.rfind("\n\n", min_boundary, end),
                text.rfind(". ", min_boundary, end),
                text.rfind("? ", min_boundary, end),
                text.rfind("! ", min_boundary, end),
                text.rfind("\n", min_boundary, end),
                text.rfind(" ", min_boundary, end),
            ]

            boundary = max(candidates)

            if boundary >= min_boundary:
                # Keep sentence punctuation, but don't include the
                # following space as part of the boundary.
                end = boundary + 1

        if end <= start:
            end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # Retain context while ensuring the next chunk starts
        # at a word boundary.
        next_start = max(start + 1, end - overlap)

        while (
            next_start < text_length
            and next_start > 0
            and text[next_start - 1].isalnum()
            and text[next_start].isalnum()
        ):
            next_start += 1

        start = next_start

    return chunks



# ============================================================
# DOCUMENT PROCESSING
# ============================================================

async def process_upload(file: UploadFile):
    filename = Path(file.filename or "").name

    if not filename or Path(filename).suffix.lower() not in {
        ".pdf",
        ".txt",
    }:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 20 MB limit.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        raw_text, page_count = extract_text(filename, content)

    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="The TXT file must use UTF-8 encoding.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this file.",
        ) from exc

    cleaned = clean_text(raw_text)

    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail=(
                "No extractable text was found. "
                "Scanned PDFs may require OCR."
            ),
        )

    chunks = chunk_text(cleaned)

    return {
        "filename": filename,
        "page_count": page_count,
        "character_count": len(cleaned),
        "content_hash": hashlib.sha256(content).hexdigest(),
        "text": cleaned,
        "chunk_count": len(chunks),
        "chunks": [
            {"chunk_id": index, "text": chunk}
            for index, chunk in enumerate(chunks, start=1)
        ],
    }


# ============================================================
# SEMANTIC SEARCH INDEX CACHE
# ============================================================

semantic_engine = SemanticSearch()
semantic_index_signature = None


def invalidate_semantic_index():
    global semantic_index_signature
    semantic_index_signature = None


# ============================================================
# SAVE OR REFRESH DOCUMENT AND CHUNKS
# ============================================================

def save_document(result: dict) -> tuple[int, bool]:
    """
    Save a new document or refresh an existing document with the
    same filename and content hash.

    Returns (document_id, created_new_document).
    Existing chunks are replaced so improved chunking is applied
    when the same document is uploaded again.
    """
    with get_connection() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM documents
            WHERE filename = ? AND content_hash = ?
            LIMIT 1
            """,
            (
                result["filename"],
                result["content_hash"],
            ),
        ).fetchone()

        if existing is not None:
            document_id = existing["id"]

            conn.execute(
                "DELETE FROM chunks WHERE document_id = ?",
                (document_id,),
            )

            conn.execute(
                """
                UPDATE documents
                SET page_count = ?,
                    character_count = ?,
                    chunk_count = ?,
                    content_hash = ?
                WHERE id = ?
                """,
                (
                    result["page_count"],
                    result["character_count"],
                    result["chunk_count"],
                    result["content_hash"],
                    document_id,
                ),
            )

            conn.executemany(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_index,
                    text
                )
                VALUES (?, ?, ?)
                """,
                [
                    (
                        document_id,
                        chunk["chunk_id"],
                        chunk["text"],
                    )
                    for chunk in result["chunks"]
                ],
            )

            created = False

        else:
            cursor = conn.execute(
                """
                INSERT INTO documents (
                    filename,
                    page_count,
                    character_count,
                    chunk_count,
                    content_hash
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result["filename"],
                    result["page_count"],
                    result["character_count"],
                    result["chunk_count"],
                    result["content_hash"],
                ),
            )

            document_id = cursor.lastrowid

            conn.executemany(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_index,
                    text
                )
                VALUES (?, ?, ?)
                """,
                [
                    (
                        document_id,
                        chunk["chunk_id"],
                        chunk["text"],
                    )
                    for chunk in result["chunks"]
                ],
            )

            created = True

    invalidate_semantic_index()

    return document_id, created


# ============================================================
# SEARCH HELPERS
# ============================================================

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_stored_chunks() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                chunks.document_id,
                documents.filename,
                chunks.chunk_index,
                chunks.text
            FROM chunks
            JOIN documents
                ON documents.id = chunks.document_id
            ORDER BY chunks.document_id, chunks.chunk_index
            """
        ).fetchall()

    return [
        {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "chunk_id": row["chunk_index"],
            "text": row["text"],
        }
        for row in rows
    ]


def get_semantic_results(
    chunks: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    global semantic_index_signature

    signature = tuple(
        (
            chunk["document_id"],
            chunk["chunk_id"],
            chunk["text"],
        )
        for chunk in chunks
    )

    if signature != semantic_index_signature:
        semantic_engine.build_index(chunks)
        semantic_index_signature = signature

    return semantic_engine.search(query, top_k)


def get_keyword_results(
    chunks: list[dict],
    query: str,
    top_k: int,
) -> list[dict]:
    query_tokens = tokenize(query)

    if not query_tokens or not chunks:
        return []

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if not any(tokenized_chunks):
        return []

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        if score <= 0:
            continue

        results.append({
            **chunks[index],
            "score": round(score, 4),
        })

        if len(results) >= top_k:
            break

    return results


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_ai_answer(
    query: str,
    results: list[dict],
) -> str:
    if not results:
        return (
            "I couldn't find relevant information "
            "in the uploaded documents."
        )

    if groq_client is None:
        return (
            "AI answer generation is unavailable because "
            "GROQ_API_KEY is not configured. "
            "Please refer to the retrieved source passages below."
        )

    context = "\n\n".join(
        (
            f"Source: {item['filename']}, "
            f"Chunk {item['chunk_id']}\n"
            f"Content: {item['text'][:2500]}"
        )
        for item in results[:5]
    )

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise document assistant. "
                        "Answer the user's question using only the "
                        "provided document passages. Give a clear, "
                        "grammatically complete answer. Use bullet "
                        "points when helpful. Do not invent facts. "
                        "If the evidence is insufficient, say so. "
                        "Cite supporting sources using the filename "
                        "and chunk number."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Retrieved document passages:\n{context}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )

        answer = response.choices[0].message.content

        return answer or "The AI could not generate an answer."

    except Exception as exc:
        print(
            f"Groq error type: {type(exc).__name__}"
        )
        print(f"Groq answer generation failed: {exc}")

        return (
            "AI answer generation is temporarily unavailable. "
            "Please refer to the retrieved source passages below."
        )


# ============================================================
# HOME AND HEALTH
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Enterprise Knowledge Retrieval Portal is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "groq_configured": groq_client is not None,
    }


# ============================================================
# EXTRACT A SINGLE DOCUMENT
# ============================================================

@app.post("/documents/extract")
async def extract_document(file: UploadFile = File(...)):
    try:
        result = await process_upload(file)

        return {
            "filename": result["filename"],
            "page_count": result["page_count"],
            "character_count": result["character_count"],
            "text": result["text"],
        }

    finally:
        await file.close()


# ============================================================
# PROCESS A SINGLE DOCUMENT
# ============================================================

@app.post("/documents/process")
async def process_document(file: UploadFile = File(...)):
    try:
        result = await process_upload(file)
        document_id, created = save_document(result)

        return {
            "document_id": document_id,
            "filename": result["filename"],
            "page_count": result["page_count"],
            "character_count": result["character_count"],
            "chunk_count": result["chunk_count"],
            "chunks": result["chunks"],
            "message": (
                "Document processed and saved successfully."
                if created
                else (
                    "Existing document refreshed with the latest "
                    "text extraction and chunking."
                )
            ),
        }

    finally:
        await file.close()


# ============================================================
# PROCESS MULTIPLE DOCUMENTS
# ============================================================

@app.post("/documents/batch-process")
async def process_multiple_documents(
    files: list[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(
            status_code=400,
            detail="Select at least one document.",
        )

    if len(files) > MAX_BATCH_FILES:
        for file in files:
            await file.close()

        raise HTTPException(
            status_code=400,
            detail=f"Upload no more than {MAX_BATCH_FILES} files per batch.",
        )

    results = []
    total_bytes = 0
    successful = 0
    failed = 0

    try:
        for file in files:
            filename = Path(file.filename or "unknown").name

            try:
                result = await process_upload(file)

                extracted_size = len(
                    result["text"].encode("utf-8")
                )

                if total_bytes + extracted_size > MAX_BATCH_SIZE:
                    failed += 1

                    results.append({
                        "filename": filename,
                        "status": "failed",
                        "message": (
                            "Batch extracted text exceeds the "
                            "100 MB safety limit."
                        ),
                    })

                    continue

                total_bytes += extracted_size

                document_id, created = save_document(result)
                successful += 1

                results.append({
                    "filename": filename,
                    "document_id": document_id,
                    "chunk_count": result["chunk_count"],
                    "status": "success",
                    "duplicate": not created,
                    "message": (
                        "Document processed and saved."
                        if created
                        else "Existing document refreshed."
                    ),
                })

            except HTTPException as exc:
                failed += 1

                results.append({
                    "filename": filename,
                    "status": "failed",
                    "message": str(exc.detail),
                })

            except Exception as exc:
                traceback.print_exc()
                failed += 1

                results.append({
                    "filename": filename,
                    "status": "failed",
                    "message": f"{type(exc).__name__}: {exc}",
                })

            finally:
                await file.close()

    finally:
        for file in files:
            if not file.file.closed:
                await file.close()

    return {
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "results": results,
        "message": "Batch processing completed.",
    }


# ============================================================
# LIST STORED DOCUMENTS
# ============================================================

@app.get("/documents")
def list_documents():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                filename,
                page_count,
                character_count,
                chunk_count
            FROM documents
            ORDER BY id DESC
            """
        ).fetchall()

    return {
        "documents": [dict(row) for row in rows]
    }


# ============================================================
# GET CHUNKS FOR A DOCUMENT
# ============================================================

@app.get("/documents/{document_id}/chunks")
def get_document_chunks(document_id: int):
    with get_connection() as conn:
        document = conn.execute(
            "SELECT id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        rows = conn.execute(
            """
            SELECT chunk_index AS chunk_id, text
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (document_id,),
        ).fetchall()

    return {
        "document_id": document_id,
        "chunks": [dict(row) for row in rows],
    }


# ============================================================
# HYBRID SEARCH: BM25 + SEMANTIC SEARCH + RRF
# ============================================================

@app.post("/search/hybrid")
def hybrid_search(request: KeywordSearchRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    chunks = load_stored_chunks()

    if not chunks:
        return {
            "query": query,
            "result_count": 0,
            "ai_answer": (
                "No documents have been uploaded yet."
            ),
            "results": [],
            "message": "No documents have been uploaded yet.",
        }

    candidate_count = min(request.top_k * 3, 50)

    keyword_results = get_keyword_results(
        chunks,
        query,
        candidate_count,
    )

    semantic_results = get_semantic_results(
        chunks,
        query,
        candidate_count,
    )

    semantic_scores = {
        (item["document_id"], item["chunk_id"]):
            float(item.get("score", 0.0))
        for item in semantic_results
    }

    # Reciprocal Rank Fusion combines the two ranked result lists.
    fused_scores = {}

    for rank, item in enumerate(keyword_results, start=1):
        key = (item["document_id"], item["chunk_id"])

        fused_scores[key] = (
            fused_scores.get(key, 0.0)
            + 1.0 / (60 + rank)
        )

    for rank, item in enumerate(semantic_results, start=1):
        key = (item["document_id"], item["chunk_id"])

        fused_scores[key] = (
            fused_scores.get(key, 0.0)
            + 1.0 / (60 + rank)
        )

    chunk_lookup = {
        (chunk["document_id"], chunk["chunk_id"]): chunk
        for chunk in chunks
    }

    ranked_results = []

    for key, rrf_score in fused_scores.items():
        chunk = chunk_lookup.get(key)

        if chunk is None:
            continue

        ranked_results.append({
            **chunk,
            "rrf_score": round(rrf_score, 6),
            "semantic_score": round(
                semantic_scores.get(key, 0.0), 4
            ),
        })

    ranked_results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    results = ranked_results[:request.top_k]
    ai_answer = generate_ai_answer(query, results)

    return {
        "query": query,
        "result_count": len(results),
        "ai_answer": ai_answer,
        "results": results,
        "message": (
            "Search completed."
            if results
            else "No relevant document chunks were found."
        ),
    }