
from pathlib import Path
import sys
import time

from langchain_community.vectorstores import FAISS

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
INDEXES_PATH = PROJECT_ROOT / "indexes"

# Allow imports from the src folder
sys.path.insert(0, str(SRC_PATH))

from embeddings import load_embedding_model


EXPERIMENTS = {
    "Fixed 500": INDEXES_PATH / "fixed_500",
    "Recursive 500 / 100": INDEXES_PATH / "recursive_500_100",
    "Recursive 1000 / 150": INDEXES_PATH / "recursive_1000_150",
}

TEST_QUESTIONS = [
    "What preventive health services are covered?",
    "What does the document say about TMJ disorder treatment?",
    "What benefit limits may apply to covered services?",
]

TOP_K = 4


def load_experiment_index(index_path, embedding_model):
    """Load one saved FAISS index."""
    faiss_file = index_path / "index.faiss"
    pickle_file = index_path / "index.pkl"

    if not faiss_file.exists() or not pickle_file.exists():
        raise FileNotFoundError(
            f"Index files are missing from: {index_path}"
        )

    return FAISS.load_local(
        str(index_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def compare_experiments():
    """Compare retrieval time and results for each index."""
    embedding_model = load_embedding_model()
    summaries = []

    for experiment_name, index_path in EXPERIMENTS.items():
        print("\n" + "=" * 65)
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Index path: {index_path}")

        try:
            vector_store = load_experiment_index(
                index_path, embedding_model
            )
        except FileNotFoundError as error:
            print(f"SKIPPED: {error}")
            continue

        experiment_times = []

        for question in TEST_QUESTIONS:
            start_time = time.perf_counter()

            results = vector_store.similarity_search(
                query=question,
                k=TOP_K,
            )

            elapsed = time.perf_counter() - start_time
            experiment_times.append(elapsed)

            print(f"\nQuestion: {question}")
            print(f"Retrieval time: {elapsed:.4f} seconds")
            print(f"Results returned: {len(results)}")

            if results:
                best_result = results[0]
                page = best_result.metadata.get("page")
                page_number = (
                    page + 1 if isinstance(page, int) else "Unknown"
                )

                print(f"Top result page: {page_number}")
                print("Top result preview:")
                print(best_result.page_content[:300])
            else:
                print("No results returned.")

        average_time = (
            sum(experiment_times) / len(experiment_times)
            if experiment_times
            else 0
        )

        summaries.append(
            (experiment_name, len(experiment_times), average_time)
        )

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("-" * 65)

    if not summaries:
        print("No experiments could be completed.")
        return

    print(f"{'Experiment':<25} {'Tests':>8} {'Avg seconds':>15}")

    for name, test_count, average_time in summaries:
        print(
            f"{name:<25} {test_count:>8} {average_time:>15.4f}"
        )

    print(
        "\nNote: Faster retrieval does not necessarily mean better "
        "relevance or answer accuracy."
    )


if __name__ == "__main__":
    compare_experiments()
