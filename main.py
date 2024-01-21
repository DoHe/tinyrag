import sys

from rag import TinyRAG


def main():
    query = "When was Arnis founded?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    print("Starting RAG...")
    rag = TinyRAG()
    print(f"Querying for '{query}'...")
    response = rag.query(query)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
