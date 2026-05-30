# Testing RAG System Context Accuracy

#### How do we know if our RAG system is working fine and has the right context to answer the user query?

#### Use below PROMPTs in succession

##### Prompt-1
```
I'm testing my RAG system, I will give you the following information: 
User Query:
Context:

I want you to tell me if the context is good and has the answers to the user's question in 1-2 lines. 

If you understand this, just say "Okay"
```

##### Prompt-2
Just Paste the output from the `retrieval_pipeline.py` code
