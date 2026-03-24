# RAG (Retrieval‑Augmented Generation) POC
This is a small personal experiment to explore how a RAG setup works using plain text documents. The idea is simple: instead of relying only on what an LLM already knows, we give it extra context pulled from your own files. That way, the model can answer questions using information that actually matters for your project.
This POC focuses on taking raw text, breaking it into useful pieces, embedding it, storing it, and then pulling the right chunks back out when a user asks something.

## 🚀 What This POC Does

- Splits raw documents into sections
- Breaks those sections into chunks
- Turns chunks into embedding vectors
- Stores everything inside a vector database
- Converts user prompts into query embeddings
- Performs semantic similarity search to find matching chunks
- Builds a context‑rich prompt for the LLM
- Generates a grounded answer based on your actual content

## Interesting Bits
[Text Ingest File Chunking](https://github.com/aam134134/ai-rag-text-vector/blob/7afefe55f518d57ce445cc0069261502dfe6446b/text-ingest/file_watcher.py#L38-L47)

[Text Chunk Embedding](https://github.com/aam134134/ai-rag-text-vector/blob/26b7ec9d05ede579e7fd596d24a8056f0870c770/chunk-embedding/embed_chunks.py#L43-L55)

## 🔄 RAG Workflow (High‑Level Overview)
1. Document Processing

   - Raw text is split into meaningful sections, keeping natural boundaries intact.
   - These sections are then chunked into smaller, retrieval‑friendly pieces.
1. Embedding
   
   - Each chunk is converted into a high‑dimensional vector that represents its semantic meaning.
   - All embeddings are stored in a vector store for fast similarity lookups.
1. Retrieval

   - When a user asks a question, the prompt is converted into a query embedding.
   - A nearest‑neighbor search finds the chunks most related to the user's intent.
1. LLM Response
   - The retrieved chunks are added to the LLM prompt as context.
   - The model responds using both its own knowledge and the actual content of your documents—producing a more accurate, grounded result.

## 🏗️ Component View — System Architecture
```mermaid
flowchart TD
    subgraph Ingestion
        D1[Text Documents]
        P1[Partitioner]
        C1[Chunker]
        E1[Embedding Service]
    end

    subgraph Retrieval
        V1["(Vector DB / Index)"]
        Q1[Query Embedding]
        R1[Similarity Search]
    end

    subgraph Generation
        L1[LLM]
        GP[Grounded Prompt Builder]
        A1[Answer]
    end

    D1 --> P1 --> C1 --> E1 --> V1
    Q1 --> R1
    R1 --> V1
    V1 --> GP
    GP --> L1 --> A1

    %% Styling
    classDef ingest fill:#fff3e0,stroke:#fb8c00,stroke-width:1px,color:#5d4037;
    classDef retr fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px,color:#0d47a1;
    classDef gen fill:#e8f5e9,stroke:#43a047,stroke-width:1px,color:#1b5e20;

    class D1,P1,C1,E1 ingest;
    class V1,Q1,R1 retr;
    class L1,GP,A1 gen;
```
