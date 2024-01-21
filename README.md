# Setup

Download phi-2 in GGUF from [TheBloke](https://huggingface.co/TheBloke/phi-2-GGUF) and put it in the parent directory of where you cloned this code.

Put documents as `.txt` into the [corpus](./corpus) directory (and delete the examples).

Install all dependencies using [poetry](https://python-poetry.org/):

    poetry install

# Running

Just run the `main.py` followed by your query, e.g.:

    poetry run python main.py 'what is the populaton of arnis?'

This will download the necessary embeddings model, start a vector database, index all documents and run your query against phi-2 with a RAG prompt.