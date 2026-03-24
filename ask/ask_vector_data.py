import argparse
import json
import os
import urllib.request

from query_vector_data import COLLECTION_NAME, query_collection

SYSTEM_PROMPT = """You answer questions using only the supplied context.
If the context is insufficient, say so clearly.
Be concise and cite the chunk numbers you used in square brackets like [1] or [2]."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Chroma, build a grounded prompt, and ask an LLM for an answer."
    )
    parser.add_argument("query", help="Natural-language question to answer.")
    parser.add_argument("--host", default="localhost", help="Chroma host.")
    parser.add_argument("--port", type=int, default=8000, help="Chroma port.")
    parser.add_argument(
        "--collection",
        default=COLLECTION_NAME,
        help=f"Collection name. Defaults to {COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to send to the LLM.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "llama3.1:8b"),
        help="Ollama model name. Defaults to env LLM_MODEL or llama3.1:8b.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
        help="Ollama generate endpoint URL.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks before the final answer.",
    )
    return parser.parse_args()


def build_context(matches):
    sections = []
    for idx, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        header = f"[{idx}] chunk_id={match.get('chunk_id')} metadata={metadata}"
        sections.append(f"{header}\n{match.get('text', '').strip()}")
    return "\n\n".join(sections)


def ask_ollama(model, ollama_url, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }
    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("response", "").strip()


def main():
    args = parse_args()

    matches = query_collection(
        query=args.query,
        host=args.host,
        port=args.port,
        collection_name=args.collection,
        top_k=args.top_k,
    )

    if not matches:
        print("No matching chunks found.")
        return

    context = build_context(matches)

    if args.show_context:
        print("Retrieved context:\n")
        print(context)
        print("\n---\n")

    user_prompt = (
        f"Question: {args.query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer the question using only the context above."
    )

    print(ask_ollama(args.model, args.ollama_url, user_prompt))


if __name__ == "__main__":
    main()
