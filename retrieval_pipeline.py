from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

PERSISTENT_DIRECTORY = "db/chromadb"
QUERY = "Who succeeded Ze'ev Drori as CEO in October 2008?"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"


def load_environment() -> None:
    """Load environment variables from a .env file."""
    load_dotenv()


def create_embedding_model(model_name: str) -> OpenAIEmbeddings:
    """Create the OpenAI embedding model."""
    return OpenAIEmbeddings(model=model_name, dimensions=1024)


def init_vector_store(persist_directory: str, embedding_model: OpenAIEmbeddings) -> Chroma:
    """Initialize the Chroma vector store."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


def create_retriever(db: Chroma, k: int = 3):
    """Build a retriever for the vector store."""
    # The retriever will return the top k most relevant documents based on cosine similarity.
    return db.as_retriever(search_kwargs={"k": k})
    # Optional: Set a score threshold for relevance
    # return db.as_retriever(search_kwargs={"k": k, "score_threshold": 0.3})

def display_documents(query: str, documents) -> None:
    """Print the query and retrieved document contents."""
    print(f"User Query: {query}")
    print("--- Context ---")
    for index, doc in enumerate(documents, 1):
        print(f"Document {index}:\n{doc.page_content}\n")
        print(f"-" * 50)


def main() -> None:
    load_environment()

    embedding_model = create_embedding_model(EMBEDDING_MODEL_NAME)
    db = init_vector_store(PERSISTENT_DIRECTORY, embedding_model)
    retriever = create_retriever(db)

    relevant_docs = retriever.invoke(QUERY)
    display_documents(QUERY, relevant_docs)


if __name__ == "__main__":
    main()


# Synthetic Questions: 

# "What was Microsoft's first hardware product release?"
# "How much did Microsoft pay to acquire GitHub?"
# "In what year did Tesla begin production of the Roadster?"
# "Who succeeded Ze'ev Drori as CEO in October 2008?"
# "What was the original name of Microsoft before it became Microsoft?"
# "How much did Microsoft pay to acquire GitHub?"