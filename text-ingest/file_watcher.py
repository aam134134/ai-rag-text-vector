import json
import uuid
import os
import time
import shutil
from pathlib import Path
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text

PROJECT_DATA_DIR = Path("./data/ingest")
WATCH_DIR = Path(os.getenv("WATCH_DIR", PROJECT_DATA_DIR / "inbox"))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", PROJECT_DATA_DIR / "archive"))
CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", PROJECT_DATA_DIR / "chunks"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

WATCH_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Watching {WATCH_DIR} -> Archiving to {ARCHIVE_DIR} -> Chunking to {CHUNKS_DIR} (every {POLL_INTERVAL}s)")

while True:
    for file in WATCH_DIR.iterdir():

        doc_id = file.stem    
        tmp = CHUNKS_DIR / f"{doc_id}.jsonl.tmp"    
        out = CHUNKS_DIR / f"{doc_id}.jsonl"

        # Read file (text)
        print("Partitioning...")
        elements = partition_text(filename=str(file))
        print("Partitioning complete.")
        #print(elements)

        print("Chunking...")
        # smaller chunks around 700-900 characters are more likely to keep one scene or exchange together
        # overlap=120 helps retrieval when a character name or idea falls near a chunk boundary
        # combine_text_under_n_chars=200 helps avoid tiny fragments like short quoted lines becoming standalone chunks
        chunks = chunk_by_title(
            elements,
            max_characters=900,
            new_after_n_chars=700,
            overlap=120,
            combine_text_under_n_chars=200,
        )

        print("Chunking complete.")
        #print(chunks)

        # Move to archive
        dest = ARCHIVE_DIR / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file.name} to archive: {dest}")


        # Write chunks as JSONL atomically
        with open(tmp, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                text = (getattr(ch, "text", "") or "").strip()

                # skip small chunks
                if len(text) < 20:
                    continue
                md = getattr(ch, "metadata", {}) or {}

                rec = {
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "text": text,
                    #"page_number": md.get("page_number"),
                    "chunk_index": i,
                    "token_count": len(text.split()),
                    "schema_version": "1.0",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Atomic finalize: rename tmp → final
        tmp.rename(out)
        print(f"[chunker] wrote {out}")    

    time.sleep(POLL_INTERVAL)