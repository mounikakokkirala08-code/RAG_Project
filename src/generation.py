
from langchain_ollama import ChatOllama
from retrieval import retrieve_chunks

MODEL_NAME = "gemma3:4b"

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


def generate_answer(question, k=4):
    """Generate an answer using retrieved SBC document passages."""

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    # Step 1: Retrieve relevant document chunks
    documents = retrieve_chunks(question, k=k)

    if not documents:
        return (
            "The SBC document does not provide enough information "
            "to answer this question."
        )

    # Step 2: Prepare context and source page numbers
    context_parts = []
    source_pages = set()

    for document in documents:
        page = document.metadata.get("page")

        if isinstance(page, int):
            page_number = page + 1
            source_pages.add(page_number)
            source = f"Source page: {page_number}"
        else:
            source = "Source page: Unknown"

        context_parts.append(
            f"{source}\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Build the prompt
    prompt = f"""
You are an assistant that answers questions about a health
plan using its Summary of Benefits and Coverage (SBC) document.

Follow these rules:
1. Answer using only the supplied document context.
2. Do not invent benefits, coverage rules, or dollar amounts.
3. If the context does not contain enough information, say:
   "The provided SBC excerpts do not contain enough information
   to answer this question."
4. Explain the answer in simple English.
5. Include source page numbers in your answer when possible.
6. Distinguish covered services from the amount a member must pay.

Document context:
{context}

User question:
{question}

Answer:
"""

    # Step 4: Generate the answer with Ollama
    response = llm.invoke(prompt)
    answer = response.content.strip()

    # Step 5: Return answer and sources
    pages_text = (
        ", ".join(str(page) for page in sorted(source_pages))
        if source_pages
        else "Unknown"
    )

    return f"{answer}\n\nSource pages: {pages_text}"


if __name__ == "__main__":
    question = "What services and benefits does this plan cover?"

    print("\nGenerating answer...")
    answer = generate_answer(question)

    print("\n--- Generated Answer ---")
    print(answer)
