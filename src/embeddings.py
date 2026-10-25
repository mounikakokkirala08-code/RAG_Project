
from chunking import create_chunks
from load_document import load_sbc_document
from langchain_huggingface import HuggingFaceEmbeddings


# Pre-trained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load a pre-trained Sentence Transformers embedding model.
    """

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    print("Embedding model loaded successfully!")

    return embeddings


def create_embeddings(chunks, embeddings):
    """
    Convert text chunks into numerical vectors.
    """

    if not chunks:
        raise ValueError("No chunks available to embed.")

    texts = [chunk.page_content for chunk in chunks]

    print(f"\nCreating embeddings for {len(texts)} chunks...")

    vectors = embeddings.embed_documents(texts)

    print("Embeddings created successfully!")
    print(f"Total vectors: {len(vectors)}")
    print(f"Vector dimensions: {len(vectors[0])}")

    return vectors


if __name__ == "__main__":

    # Step 1: Load the SBC PDF
    pages = load_sbc_document()

    # Step 2: Split the PDF into chunks
    chunks = create_chunks(
        pages,
        chunk_size=500,
        chunk_overlap=100,
        strategy="recursive",
    )

    # Step 3: Load the embedding model
    embedding_model = load_embedding_model()

    # Step 4: Convert chunks into vectors
    vectors = create_embeddings(chunks, embedding_model)

    # Display the first vector's first 10 numbers
    print("\nFirst 10 values of the first vector:")
    print(vectors[0][:10])


#.3
# what are embedding provided by goggle or azure
# types of  embedding models
# dimentions are sematic meaning of input
# 766 dimentions tells about the particular object or word
# explain what vector and dimention