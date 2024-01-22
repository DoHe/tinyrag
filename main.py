from rag import TinyRAG


def main():
    print("Starting RAG...")
    rag = TinyRAG()

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
    main()
