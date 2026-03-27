import argparse
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "ai-rag-text-vector"
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or None
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "true").lower() == "true"
QUERY_TEXT_PREFIX = os.getenv("QUERY_TEXT_PREFIX", "Represent this sentence for searching relevant passages: ")


def build_query_input(query):
    return f"{QUERY_TEXT_PREFIX}{query}" if QUERY_TEXT_PREFIX else query


def embed_query(query):
    model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    return model.encode(
        [build_query_input(query)],
        convert_to_numpy=True,
        normalize_embeddings=EMBED_NORMALIZE,
        show_progress_bar=False,
    )[0].tolist()


def query_collection(
    query,
    host=VECTOR_DB_HOST,
    port=VECTOR_DB_PORT,
    collection_name=COLLECTION_NAME,
    top_k=TOP_K,
):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_embedding = embed_query(query)
    client = chromadb.HttpClient(host=host, port=port)
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    matches = []
    for idx, document in enumerate(documents):
        matches.append(
            {
                "chunk_id": ids[idx] if idx < len(ids) else None,
                "distance": distances[idx] if idx < len(distances) else None,
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "text": document,
            }
        )

    return matches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the Chroma vector store for the most relevant text chunks."
    )
    parser.add_argument("query", help="Natural-language query to search for.")
    return parser.parse_args()


def main():
    args = parse_args()

    matches = query_collection(
        query=args.query,
        host=VECTOR_DB_HOST,
        port=VECTOR_DB_PORT,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
    )

    if not matches:
        print("No matching chunks found.")
        return

    for idx, match in enumerate(matches, start=1):
        print(f"Result {idx}")
        print(f"chunk_id: {match['chunk_id']}")
        if match["distance"] is not None:
            print(f"distance: {match['distance']}")
        print(f"metadata: {json.dumps(match['metadata'], ensure_ascii=False)}")
        print("text:")
        print(match["text"])
        print()


if __name__ == "__main__":
    main()
