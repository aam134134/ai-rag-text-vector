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


def build_query_input(query):
    return f"{QUERY_TEXT_PREFIX}{query}" if QUERY_TEXT_PREFIX else query


def main():
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    query_embedding = model.encode(
        [build_query_input(args.query)],
        convert_to_numpy=True,
        normalize_embeddings=EMBED_NORMALIZE,
        show_progress_bar=False,
    )[0].tolist()

    client = chromadb.HttpClient(host=args.host, port=args.port)
    collection = client.get_collection(name=args.collection)

    include_fields = ["documents", "metadatas", "distances"]
    if args.include_embeddings:
        include_fields.append("embeddings")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.top_k,
        include=include_fields,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not documents:
        print("No matching chunks found.")
        return

    for idx, document in enumerate(documents, start=1):
        metadata = metadatas[idx - 1] if idx - 1 < len(metadatas) else {}
        distance = distances[idx - 1] if idx - 1 < len(distances) else None
        chunk_id = ids[idx - 1] if idx - 1 < len(ids) else None

        print(f"Result {idx}")
        print(f"chunk_id: {chunk_id}")
        if distance is not None:
            print(f"distance: {distance}")
        print(f"metadata: {json.dumps(metadata, ensure_ascii=False)}")
        print("text:")
        print(document)
        print()


if __name__ == "__main__":
    main()
