import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain import hub


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def log_chain(args, message):
    print(message)
    return args


class TinyRAG:
    def __init__(
        self,
        model_path,
        quantization,
        embedding_model,
        corpus_directory,
        num_retrieved,
        chunk_size,
        chunk_overlap,
    ):
        self.model_path = model_path
        self.quantization = quantization
        self.embedding_model = embedding_model
        self.corpus_directory = corpus_directory
        self.num_retrieved = num_retrieved
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        splits = self._load_docs()
        embedder = self._load_embeddings()
        retriever = self._start_vectore_store(documents=splits, embedding=embedder)
        llm = self._load_llm()

        self.rag_chain, self.non_rag_chain = self._assemble_chains(retriever, llm)

    def _load_docs(self):
        print("Loading documents...")
        loader = DirectoryLoader(
            self.corpus_directory,
            glob="*.txt",
            use_multithreading=True,
            show_progress=True,
            loader_cls=TextLoader,
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )

        print("Splitting docs...")
        splits = text_splitter.split_documents(docs)
        print(f"Have {len(splits)} splits")

        return splits

    def _load_embeddings(self):
        print("Loading embeddings model...")
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embedder = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print("Loaded")

        return embedder

    def _start_vectore_store(self, documents, embedding):
        print("Creating vector store...")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embedding)
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.num_retrieved}
        )
        print("Done")

        return retriever

    def _load_llm(self):
        print("Loading LLM...")
        file_name = f"phi-2.{self.quantization}.gguf"
        llm = LlamaCpp(
            model_path=os.path.join(self.model_path, file_name),
            temperature=0.1,
            max_tokens=1000,
            top_p=1,
        )
        print("Done")

        return llm

    def _assemble_chains(self, retriever, llm):
        print("Assembling chains...")
        rag_prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            RunnableLambda(lambda args: log_chain(args, "Retrieving context"))
            | {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnableLambda(lambda args: log_chain(args, "Got context"))
            | rag_prompt
            | RunnableLambda(lambda args: log_chain(args, "Prompting LLM"))
            | llm
            | StrOutputParser()
        )
        non_rag_prompt = PromptTemplate.from_template(
            "You are an assistant for question-answering tasks. If you don't know the answer, "
            "just say that you don't know. Use three sentences maximum and keep the answer concise."
            "\nQuestion: {question} \nAnswer:"
        )
        non_rag_chain = (
            {"question": RunnablePassthrough()}
            | non_rag_prompt
            | RunnableLambda(lambda args: log_chain(args, "Prompting LLM"))
            | llm
            | StrOutputParser()
        )
        print("Done")

        return rag_chain, non_rag_chain

    def query(self, query: str, use_rag=True) -> str:
        chain = self.rag_chain if use_rag else self.non_rag_chain
        chunks = ""
        for chunk in chain.stream(query):
            print(chunk, end="", flush=True)
            chunks += chunk
        return chunks
