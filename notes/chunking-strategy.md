# Chunking Strategies

## Why Chunking Matters
The quality of RAG output depends heavily on how documents are split. Basic methods like Character Text Splitter often result in:

- Mid-sentence splits, which break the flow of information.
- Loss of context by separating related concepts across different chunks.
- Poor retrieval quality, as the system searches chunks rather than entire documents.

---

##  Strategies:

- **Character Text Splitter**: Basic, useful for simple/uniform documents where speed is the priority.
- **Recursive Character Text Splitter**: An upgrade that respects natural boundaries (paragraphs, sentences, words) and preserves context better than basic splitting.
- **Document-Specific Splitting**: Tailored to the file format (e.g., PDFs, Markdown, CSVs), maintaining awareness of headers, pages, and sections.
- **Semantic Splitting**: Uses embeddings to detect topic shifts, ensuring related concepts stay together; more intelligent but computationally expensive.
- **Agentic/AI-Powered Chunking** : Uses an LLM to analyze content and determine optimal, context-aware splits; the most sophisticated yet slowest and most expensive method.

---
