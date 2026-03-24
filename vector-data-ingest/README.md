# Insert Vector Data

## To Build and Run via Podman/Docker

- Build and run: ```podman compose up --build```
- Drop a JSONL (*.jsonl) file with embeds into ```../data/chunks/embeds```
- Shutdown container: ```podman compose down```

## To Setup and Run Locally

- Requires Python 3.12
- Setup the virtual environment
  - ```python3.12 -m venv .venv```
- Activate and setup the virtual environment
  - Activate the venv (choose one)
    - macOS/Linux: ```source .venv/bin/activate```
    - Windows (Command Prompt): ```.venv\Scripts\activate```
    - Windows (PowerShell): ```.venv\Scripts\Activate.ps1```
  - ```python -m pip install --upgrade pip setuptools wheel```
  - ```python -m pip install -r requirements.txt```
- Run program: ```python insert_vector_data.py```
- Drop a JSONL (*.jsonl) file with embeds into ```../data/chunks/embeds```