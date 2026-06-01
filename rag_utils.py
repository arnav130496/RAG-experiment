import os
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

DEFAULT_PERSISTENT_DIRECTORY = "db/chromadb"
DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_CHROMA_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def load_environment() -> None:
    """Load environment variables from a .env file."""
    load_dotenv()


def get_persistent_directory() -> str:
    """Return the configured Chroma persistent directory."""
    return os.environ.get("PERSISTENT_DIRECTORY", DEFAULT_PERSISTENT_DIRECTORY)


def get_embedding_model_name() -> str:
    """Return the configured OpenAI embedding model name."""
    return os.environ.get("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)


def get_embedding_dimensions() -> int:
    """Return the configured embedding dimensions."""
    return int(os.environ.get("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS))


def create_embedding_model(
    model_name: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OpenAIEmbeddings:
    """Create an OpenAI embedding model using configured defaults."""
    if model_name is None:
        model_name = get_embedding_model_name()
    if dimensions is None:
        dimensions = get_embedding_dimensions()
    return OpenAIEmbeddings(model=model_name, dimensions=dimensions)


def init_vector_store(
    persist_directory: Optional[str] = None,
    embedding_model: Optional[OpenAIEmbeddings] = None,
) -> Chroma:
    """Initialize a Chroma vector store using the configured persistent directory."""
    if persist_directory is None:
        persist_directory = get_persistent_directory()
    if embedding_model is None:
        embedding_model = create_embedding_model()
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_metadata=DEFAULT_CHROMA_COLLECTION_METADATA,
    )


def create_vector_store_from_documents(
    chunks: list,
    persist_directory: Optional[str] = None,
    embedding_model: Optional[OpenAIEmbeddings] = None,
) -> Chroma:
    """Create and persist a Chroma vector store from document chunks."""
    if persist_directory is None:
        persist_directory = get_persistent_directory()
    if embedding_model is None:
        embedding_model = create_embedding_model()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata=DEFAULT_CHROMA_COLLECTION_METADATA,
    )


def create_retriever(db: Chroma, k: int = 3, score_threshold: Optional[float] = None):
    """Create a retriever for a Chroma database."""
    search_kwargs = {"k": k}
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold
    return db.as_retriever(search_kwargs=search_kwargs)
