# Setup

Download phi-2 in GGUF from [TheBloke](https://huggingface.co/TheBloke/phi-2-GGUF) and put it in the parent directory of where you cloned this code.

Put documents as `.txt` into the [corpus](./corpus) directory (and delete the examples).

Install all dependencies using [poetry](https://python-poetry.org/):

    poetry install

# Running

Just run the `main.py` using:

    poetry run python main.py

This will download the necessary embeddings model, start a vector database, index all documents and wait for your input.
It will run your prompt against phi-2 with an optional RAG prompt and context from the vector database. If you deactivate RAG typing 'd', the prompt will directly be sent to phi-2, without a system prompt or context data.