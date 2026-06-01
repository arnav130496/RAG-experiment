import os
from chromadb import Documents
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from rag_utils import create_vector_store_from_documents, load_environment


load_environment()

def load_documents(docs_path: str = "docs")-> list[Documents]:
    """Load documents from the specified directory."""
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError("DOCUMENTS_PATH environment variable is not set.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()
    if len(documents) == 0:
        raise ValueError("No documents found in the specified directory.")
    
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")

    return documents

def split_documents(documents: list[Documents], chunk_size: int = 1000, chunk_overlap: int = 0) -> list[Documents]:
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks: list[Documents], persist_directory: str = "db/chromadb"):
    """Create a Chroma vector store from the document chunks."""
    print("Creating vector store...")
    print(f"--- Creating Chroma vector store with {len(chunks)} chunks ---")

    vector_store = create_vector_store_from_documents(
        chunks,
        persist_directory=persist_directory,
    )
    print(f"--- Finished creating Chroma vector store with {len(chunks)} chunks ---")
    print(f"Vector store created and saved in directory: {persist_directory}")

    return vector_store

def main():
    print("Hello Main!")

    #1 Load documents
    docs = load_documents(docs_path="docs")
    #2 Split documents into chunks
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=0)
    #3 Create vector store
    vector_store = create_vector_store(chunks, persist_directory="db/chromadb")

if __name__ == "__main__":
    main()