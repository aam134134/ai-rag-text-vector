import os
import time
import shutil
from pathlib import Path
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.html import partition_html
from unstructured.partition.txt import partition_text

WATCH_DIR = Path(os.getenv("WATCH_DIR", "/home/appuser/data/inbox"))
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", "/home/appuser/data/archive"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "3330"))

WATCH_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"Watching {WATCH_DIR} -> Archiving to {ARCHIVE_DIR} (every {POLL_INTERVAL}s)")

while True:
    for file in WATCH_DIR.iterdir():

        # Read file (text)
        elements = partition_text(file=file)
        print(elements)

        # Move to archive
        dest = ARCHIVE_DIR / file.name
        shutil.move(str(file), str(dest))
        print(f"Moved {file.name} to archive: {dest}")

    time.sleep(POLL_INTERVAL)

