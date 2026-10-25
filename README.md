
# SBC RAG - Health Benefits Assistant

## 1. Project Overview

SBC RAG is a Retrieval-Augmented Generation (RAG) application
that answers questions about a Summary of Benefits and Coverage
(SBC) PDF document.

The application retrieves relevant passages from the document
and uses a local Large Language Model (LLM) to generate answers.

## 2. Technologies Used

- Python
- LangChain
- PyPDF
- Sentence Transformers
- FAISS vector store
- Ollama
- Gemma 3 (4B)
- Streamlit
- uv

## 3. Project Architecture

SBC PDF
   |
   v
PDF Loading
   |
   v
Text Chunking
   |
   v
Sentence Transformer Embeddings
   |
   v
FAISS Vector Index
   |
   v
User Question
   |
   v
Similarity Search
   |
   v
Retrieved Document Context
   |
   v
Gemma 3 through Ollama
   |
   v
Answer with Source Page References

## 4. Project Structure

```text
sbc-rag/
├── data/
│   └── sbc.pdf
├── indexes/
│   ├── fixed_500/
│   ├── recursive_500_100/
│   ├── recursive_1000_150/
│   └── sbc_faiss_index/
├── src/
│   ├── load_document.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── retrieval.py
│   └── generation.py
├── evaluation/
│   ├── test_questions.json
│   └── compare_experiments.py
├── app.py
├── requirements.txt
└── README.md
```

## 5. Prerequisites

Install Python, uv, and Ollama.

Download the Gemma model if it is not already available:

```powershell
ollama pull gemma3:4b
```

## 6. Install Dependencies

Activate your project environment if needed, then run:

```powershell
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
```

## 7. Add the SBC PDF

Place your PDF at:

```text
data/sbc.pdf
```

Use a document you are authorized to process.

## 8. Build the FAISS Index

Run:

```powershell
python src/indexing.py
```

This loads the PDF, creates chunks and embeddings,
and saves the FAISS index.

## 9. Test Document Retrieval

Run:

```powershell
python src/retrieval.py
```

## 10. Test Answer Generation

Run:

```powershell
python src/generation.py
```

Make sure Ollama is running and the configured model is available.

## 11. Launch the Streamlit Application

Run:

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## 12. Compare Chunking Experiments

Run:

```powershell
python evaluation/compare_experiments.py
```

The experiment compares retrieval time and retrieved text
across the saved chunking strategies.

## 13. Important Limitations

- Answers depend on the relevance and completeness of retrieved passages.
- Source page references help locate information but do not guarantee accuracy.
- Coverage, exclusions, limits, and member costs must be verified
  against the applicable full plan documents.
- Do not use the application as the sole basis for healthcare
  coverage or financial decisions.
- FAISS index files must be loaded only from trusted sources.

## 14. Future Improvements

- Improve retrieval with more evaluation questions.
- Compare retrieval quality using verified expected results.
- Add source passage display in the user interface.
- Add document upload and re-indexing.
- Add automated tests and error handling.
- Evaluate answer faithfulness and retrieval relevance.
