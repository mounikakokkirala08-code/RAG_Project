
from pathlib import Path

from langchain_community.vectorstores import FAISS
from embeddings import load_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "indexes" / "sbc_faiss_index"


def load_faiss_index():
    """Load the saved FAISS index."""

    faiss_file = INDEX_PATH / "index.faiss"
    pickle_file = INDEX_PATH / "index.pkl"

    if not faiss_file.exists() or not pickle_file.exists():
        raise FileNotFoundError(
            f"FAISS index files not found in: {INDEX_PATH}\n"
            "Run 'python src/indexing.py' first."
        )

    embedding_model = load_embedding_model()

    vector_store = FAISS.load_local(
        str(INDEX_PATH),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    print("FAISS index loaded successfully!")
    return vector_store


def retrieve_chunks(question, k=4):
    """Retrieve the most relevant chunks for a question."""

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    vector_store = load_faiss_index()

    print(f"\nQuestion: {question}")
    print(f"Retrieving up to {k} relevant chunks...")

    results = vector_store.similarity_search(
        query=question,
        k=k,
    )

    if not results:
        print("No relevant chunks found.")
        return []

    for i, document in enumerate(results, start=1):
        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "Unknown"

        print(f"\n--- Result {i} | Page {page_number} ---")
        print(document.page_content)

    return results


if __name__ == "__main__":
    question = "What services and benefits does this plan cover?"
    documents = retrieve_chunks(question, k=4)

    print(f"\nTotal retrieved chunks: {len(documents)}")
