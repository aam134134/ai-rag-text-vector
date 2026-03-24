# Vector Database (Chroma)

Directory includes code to insert vectors and associated metadata and query this data

To start the Chroma Vector DB container:
```podman compose up -d```

To stop the Chroma Vector DB container:
```podman compose down```

To query the collection for the most relevant chunks:
```bash
python query_vector_data.py "What is this document about?"
```

Useful flags:
```bash
python query_vector_data.py "What is this document about?" --top-k 5 --include-embeddings
```

To retrieve chunks and ask an LLM for a grounded answer:
```bash
python ask_vector_data.py "Who is Anna Pavlovna Scherer?"
```

This uses a local Ollama server by default. Example with an explicit local model:
```bash
python ask_vector_data.py "Who is Anna Pavlovna Scherer?" --model llama3.1:8b
```
