
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from functools import lru_cache
from time import perf_counter

from src.chunking import STRATEGIES
from src.indexing import load_index
from src.generation import generate_answer

app = FastAPI(title="SBC RAG API")


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    strategy: Literal[
        "fixed_500",
        "recursive_500_100",
        "recursive_1000_150",
    ] = "recursive_500_100"
    top_k: int = Field(default=3, ge=1, le=8)


@lru_cache(maxsize=3)
def get_vector_store(strategy: str):
    return load_index(strategy)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/strategies")
def strategies():
    return {"strategies": STRATEGIES}


@app.post("/query")
def query_sbc(request: QueryRequest):
    try:
        store = get_vector_store(request.strategy)

        start = perf_counter()
        results = store.similarity_search_with_score(
            request.question,
            k=request.top_k,
        )
        retrieval_seconds = perf_counter() - start

        chunks = [
            {
                "text": doc.page_content,
                "page": doc.metadata.get("page"),
                "distance": float(score),
            }
            for doc, score in results
        ]

        if not chunks:
            return {
                "question": request.question,
                "answer": "No relevant content found.",
                "sources": [],
            }

        answer = generate_answer(
            question=request.question,
            retrieved_chunks=chunks,
            model="gemma3:4b",
        )

        return {
            "question": request.question,
            "answer": answer,
            "strategy": request.strategy,
            "sources": chunks,
            "retrieval_seconds": round(retrieval_seconds, 4),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Index not found. Build the index first: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
