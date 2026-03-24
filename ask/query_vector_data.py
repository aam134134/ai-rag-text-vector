import argparse
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "ai-rag-text-vector"
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("EMBED_DEVICE")
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "true").lower() == "true"
QUERY_TEXT_PREFIX = os.getenv("QUERY_TEXT_PREFIX", "")


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
    host="localhost",
    port=8000,
    collection_name=COLLECTION_NAME,
    top_k=3,
    include_embeddings=False,
):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_embedding = embed_query(query)
    client = chromadb.HttpClient(host=host, port=port)
    collection = client.get_collection(name=collection_name)

    include_fields = ["documents", "metadatas", "distances"]
    if include_embeddings:
        include_fields.append("embeddings")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=include_fields,
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
    parser.add_argument(
        "--host",
        default="localhost",
        help="Chroma host. Defaults to localhost.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Chroma port. Defaults to 8000.",
    )
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Collection name. Defaults to {COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matches to return. Defaults to 3.",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include embeddings in the raw JSON output from Chroma.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    matches = query_collection(
        query=args.query,
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        top_k=args.top_k,
        include_embeddings=args.include_embeddings
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
