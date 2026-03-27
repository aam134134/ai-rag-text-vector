import argparse
import json
import os
import urllib.request
from urllib.error import HTTPError, URLError

from query_vector_data import COLLECTION_NAME, query_collection

VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", "8000"))
TOP_K = int(os.getenv("TOP_K", "5"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

SYSTEM_PROMPT = """
You must answer using only the provided context.

Output format (do not change):
Answer: <short answer>
Evidence: <exact quote from the context, or None>

Rules:
- Do not use outside knowledge.
- Base the answer only on the context.
- If the answer is not explicitly stated in the context, write:
  Answer: The context does not provide enough information.
  Evidence: None
- The Evidence field must contain an exact supporting quote copied from the context.
- Keep the Answer short and factual.
- Do not add any extra text.
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Chroma, build a prompt, and ask an LLM for an answer."
    )
    parser.add_argument("query", help="Natural-language question to answer.")
    return parser.parse_args()


def build_context(matches):
    sections = []
    for idx, match in enumerate(matches, start=1):
        metadata = match.get("metadata") or {}
        header = f"[{idx}] chunk_id={match.get('chunk_id')} metadata={metadata}"
        sections.append(f"{header}\n{match.get('text', '').strip()}")
    return "\n\n".join(sections)


def ask_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama request failed with status {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach Ollama at {OLLAMA_URL}: {exc.reason}") from exc
    return body.get("response", "").strip()


def answer_question(query):
    matches = query_collection(
        query=query,
        host=VECTOR_DB_HOST,
        port=VECTOR_DB_PORT,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
    )

    if not matches:
        return {
            "query": query,
            "answer": "No matching chunks found.",
            "matches": [],
            "context": "",
        }

    context = build_context(matches)
    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
    )
    answer = ask_ollama(user_prompt)
    return {
        "query": query,
        "answer": answer,
        "matches": matches,
        "context": context,
    }


def main():
    args = parse_args()
    result = answer_question(args.query)
    print(result["answer"])


if __name__ == "__main__":
    main()
