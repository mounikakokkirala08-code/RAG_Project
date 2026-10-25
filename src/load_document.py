
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PDF file path
PDF_PATH = PROJECT_ROOT / "data" / "sbc.pdf"


def load_sbc_document():
    """
    Load the SBC PDF and return a list of page-level documents.
    Each document contains page text and metadata.
    """

    # Check whether the PDF exists
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"SBC PDF not found: {PDF_PATH}\n"
            "Please place your PDF at data/sbc.pdf."
        )

    # Load the PDF
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()

    # Check whether any pages were extracted
    if not pages:
        raise ValueError("The PDF contains no extractable pages.")

    # Print document information
    print("PDF loaded successfully!")
    print(f"PDF path: {PDF_PATH}")
    print(f"Total pages: {len(pages)}")

    return pages


if __name__ == "__main__":
    pages = load_sbc_document()

    # Display text from the first two pages
    for page in pages[:2]:
        page_number = page.metadata.get("page", 0) + 1

        print(f"\n--- Page {page_number} ---")
        print(page.page_content[:500])


# pypdf
# meta data

# how much long pdf or document -- 80 t0 100
# how r u dealing with table in pdf to preserve data ?

# how will u deal with images in pdf ? 
