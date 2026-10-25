
from pathlib import Path

from langchain_community.vectorstores import FAISS

from chunking import create_chunks
from load_document import load_sbc_document
from embeddings import load_embedding_model


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "indexes" / "sbc_faiss_index"


def build_faiss_index():
    """
    Load the SBC PDF, split it into chunks,
    create embeddings, and save them in a FAISS index.
    """

    # 1. Load the SBC PDF
    pages = load_sbc_document()

    # 2. Split the PDF into chunks
    chunks = create_chunks(
        pages=pages,
        chunk_size=500,
        chunk_overlap=100,
        strategy="recursive",
    )

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    # 3. Load the embedding model
    embedding_model = load_embedding_model()

    # 4. Create the FAISS index from the chunks
    print("\nBuilding FAISS index...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    # 5. Create the indexes directory if needed
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 6. Save the index locally
    vector_store.save_local(str(INDEX_PATH))

    print("\nFAISS index created successfully!")
    print(f"Saved index location: {INDEX_PATH}")
    print(f"Indexed chunks: {len(chunks)}")

    return vector_store


if __name__ == "__main__":
    build_faiss_index()
