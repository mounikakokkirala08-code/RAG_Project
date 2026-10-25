
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from load_document import load_sbc_document


def create_chunks(
    pages,
    chunk_size=500,
    chunk_overlap=100,
    strategy="recursive",
):
    """
    Split page-level documents into smaller text chunks.

    Args:
        pages: Documents returned by load_sbc_document().
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters shared between neighboring chunks.
        strategy: 'recursive' or 'fixed'.

    Returns:
        A list of chunk documents with text and metadata.
    """

    # Validate inputs
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
        )

    if strategy == "recursive":
        # Prefer splitting at natural text boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    elif strategy == "fixed":
        # Split text using spaces as separators
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    else:
        raise ValueError(
            "Invalid strategy. Choose 'recursive' or 'fixed'."
        )

    # Create chunks
    chunks = splitter.split_documents(pages)

    # Display results
    print("\n--- Chunking Results ---")
    print(f"Strategy: {strategy}")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")
    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")

    # Display first chunk and metadata
    if chunks:
        print("\n--- First Chunk ---")
        print(chunks[0].page_content)

        print("\n--- First Chunk Metadata ---")
        print(chunks[0].metadata)
    else:
        print("No chunks were created.")

    return chunks


if __name__ == "__main__":
    # Step 1: Load the PDF
    pages = load_sbc_document()

    # Step 2: Split the PDF into chunks
    chunks = create_chunks(
        pages=pages,
        chunk_size=500,
        chunk_overlap=100,
        strategy="recursive",
    )


# llm parser chunking strategy how will u do imprementation?
# RecursiveCharacterTextSplitter  why u used ?




