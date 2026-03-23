import time, shutil, chromadb, jsonlines, os
from pathlib import Path

PROJECT_DATA_DIR = Path("../data")
WATCH_DIR = Path(os.getenv("WATCH_DIR", PROJECT_DATA_DIR / "chunks/embeds"))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", PROJECT_DATA_DIR / "embeds/archive"))
VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

WATCH_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

client = chromadb.HttpClient(host=VECTOR_DB_HOST, port=8000)

# Create collection
collection = client.get_or_create_collection(
    name="ai-rag-text-vector",
    metadata={"description": "Db store for ai-rag-text-vector"}
)

print(f"Watching {WATCH_DIR} -> Archiving to {ARCHIVE_DIR} (every {POLL_INTERVAL}s)")

while True:
    for file in WATCH_DIR.glob("*.jsonl"):

        print(f"Inserting embeds for {file.name}...")

        with jsonlines.open(file, "r") as reader:
            for row in reader:
                chunk_id = row["chunk_id"]
                text = row["text"]
                embedding = row["embedding"]
                metadata = {k: v for k, v in row.items() if k not in ["embedding", "text"]}

                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[text],
                    metadatas=[metadata]
                )

        print("Embed insert complete.")

        # Move to archive
        dest = ARCHIVE_DIR / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file.name} to archive: {dest}")

    time.sleep(POLL_INTERVAL)
