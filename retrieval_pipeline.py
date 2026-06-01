from rag_utils import (
    create_embedding_model,
    create_retriever,
    init_vector_store,
    load_environment,
)

QUERY = "Who succeeded Ze'ev Drori as CEO in October 2008?"

def display_documents(query: str, documents) -> None:
    """Print the query and retrieved document contents."""
    print(f"User Query: {query}")
    print("--- Context ---")
    for index, doc in enumerate(documents, 1):
        print(f"Document {index}:\n{doc.page_content}\n")
        print(f"-" * 50)


def main() -> None:
    load_environment()

    embedding_model = create_embedding_model()
    db = init_vector_store(embedding_model=embedding_model)
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