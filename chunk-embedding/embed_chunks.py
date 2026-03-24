import os
import time
import jsonlines
import shutil
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_DATA_DIR = Path("../data")
WATCH_DIR = Path(os.getenv("WATCH_DIR", PROJECT_DATA_DIR / "ingest/chunks"))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", PROJECT_DATA_DIR / "chunks/archive"))
EMBEDS_DIR = Path(os.getenv("EMBEDS_DIR", PROJECT_DATA_DIR / "chunks/embeds"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

WATCH_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Watching {WATCH_DIR} -> Archiving to {ARCHIVE_DIR} -> Embedding to {EMBEDS_DIR} (every {POLL_INTERVAL}s)")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

while True:
    for file in WATCH_DIR.glob("*.jsonl"):

        doc_id = file.stem    
        tmp = EMBEDS_DIR / f"{doc_id}.jsonl.tmp"    
        out = EMBEDS_DIR / f"{doc_id}.jsonl"

        # Stream read input and stream write output to tmp
        count = 0
        print("Creating embeds...")

        # Open the tmp file ourselves so we can fsync before renaming
        with open(tmp, "w", encoding="utf-8") as f_out:
            writer = jsonlines.Writer(f_out)

            try:
                with jsonlines.open(file, "r") as reader:
                    for row in reader:
                        # Require a 'text' field
                        text = row.get("text")

                        # Encode ONE line at a time to keep memory low
                        # convert_to_numpy=True means the embedding is returned as a NumPy array instead of a framework-specific tensor.
                        # Useful in a local RAG pipeline because it makes the vector easy to serialize, convert to 
                        # a Python list, and store in Chroma without extra PyTorch handling.
                        # normalize_embeddings=True scales each embedding to unit length. 
                        # Helpful for semantic search because similarity comparisons become more stable 
                        # and retrieval quality is often better when document and query vectors are normalized the same way.
                        vec = model.encode(
                            [text],
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False
                        )[0]

                        # Append embedding to the row and write immediately
                        row["embedding"] = vec.tolist()
                        writer.write(row)
                        count += 1

            finally:
                # Ensure everything is flushed to disk before renaming
                writer.close()
                f_out.flush()
                os.fsync(f_out.fileno())

        # Atomically rename tmp -> final (.jsonl)
        os.replace(tmp, out)
        print(f"Embeds complete. Wrote {count} lines to: {out.absolute}")

        # Move to archive
        dest = ARCHIVE_DIR / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file.name} to archive: {dest}")

    time.sleep(POLL_INTERVAL)