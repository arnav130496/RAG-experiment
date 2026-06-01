from rag_utils import (
    create_embedding_model,
    create_retriever,
    init_vector_store,
    load_environment,
    get_persistent_directory,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_environment()

persistent_directory = get_persistent_directory()

# Load embeddings and vector store
embedding_model = create_embedding_model()

db = init_vector_store(
    persist_directory=persistent_directory,
    embedding_model=embedding_model,
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = create_retriever(db, k=5)

# If you want to require a minimum relevance score instead, pass score_threshold to create_retriever.
# retriever = create_retriever(db, k=5, score_threshold=0.3)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)