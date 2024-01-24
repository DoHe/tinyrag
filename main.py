from rag import TinyRAG

QUANTS = [
    "Q2_K",
    "Q3_K_L",
    "Q3_K_M",
    "Q3_K_S",
    "Q4_0",
    "Q4_K_M",
    "Q4_K_S",
    "Q5_0",
    "Q5_K_M",
    "Q5_K_S",
    "Q6_K",
    "Q8_0",
]


def main(
    model_path,
    quantization,
    embedding_model,
    corpus_directory,
    num_retrieved,
    chunk_size,
    chunk_overlap,
):
    print("Starting RAG...")
    rag = TinyRAG(
        model_path,
        quantization,
        embedding_model,
        corpus_directory,
        num_retrieved,
        chunk_size,
        chunk_overlap,
    )

    rag_active = True
    while True:
        activate_text = "'d' to disable rag" if rag_active else "'a' to activate rag"
        input_text = f"Write your prompt or {activate_text}:\n"
        query = input(input_text)

        if not query:
            continue
        if query == "d":
            rag_active = False
            continue
        if query == "a":
            rag_active = True
            continue

        rag.query(query, rag_active)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="tinyrag",
        description="A tiny RAG enabled LLM",
        epilog="Enjoy!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model-path",
        default="../phi-2-GGUF/",
        help="The directory containing the phi2 gguf model files",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        default="Q5_K_S",
        choices=QUANTS,
        help="Which quantization to use",
    )
    parser.add_argument(
        "-e",
        "--embedding-model",
        default="BAAI/bge-small-en",
        help="Which huggingface-hosted embeddings model to use",
    )
    parser.add_argument(
        "-c",
        "--corpus-directory",
        default="./corpus",
        help="The directory that contains the documents (as .txt files) "
        "to build the rag corpus from",
    )
    parser.add_argument(
        "-k",
        "--num-retrieved",
        default=2,
        type=int,
        help="How many documents to retrieve to be added to the context",
    )
    parser.add_argument(
        "-cs",
        "--chunk-size",
        default=400,
        type=int,
        help="The size of the chunks for text splitting documents in the corpus",
    )
    parser.add_argument(
        "-co",
        "--chunk-overlap",
        default=100,
        type=int,
        help="By how much each chunk in text splitting should overlap with the previous chunk",
    )

    args = parser.parse_args()

    main(
        args.model_path,
        args.quantization,
        args.embedding_model,
        args.corpus_directory,
        args.num_retrieved,
        args.chunk_size,
        args.chunk_overlap,
    )
