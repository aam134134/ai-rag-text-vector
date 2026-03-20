# [Unstructured](https://unstructured.io) Text Ingest Container 

[Unstructured install doc](https://docs.unstructured.io/open-source/installation/full-installation)

## To Build and Run via Podman/Docker

- Build and run: ```podman compose up --build```
- Drop a text (*.txt) file to ingest into ```./data/ingest/inbox```
- View chunked data in ```./data/chunks```

## To Setup and Run Locally

- Requires Python 3.12
- Setup the virtual environment
  - ```python3.12 -m venv .venv```
- Activate and setup the virtual environment
  - macOS/Linux: ```source venv/bin/activate```
  - Windows (Command Prompt): ```venv\Scripts\activate```
  - Windows (PowerShell): ```venv\Scripts\Activate.ps1```
  - ```python -m pip install --upgrade pip setuptools wheel```
  - ```python -m pip install -r requirements.txt```
- Run program: ```python file_watcher.py```
- Drop a text (*.txt) file to ingest into ```./data/ingest/inbox```
- View chunked data in ```./data/chunks```
