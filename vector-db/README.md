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
