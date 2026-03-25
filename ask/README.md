# Ask the LLM

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

## Query and View Vector DB Results
To query the collection for the most relevant chunks:
```bash
python query_vector_data.py "What is Anna Pavlovna's view about Russia's role in Europe?"
```

## Query and View LLM Results using Vector DB Data

```bash
python ask_vector_data.py "What is Anna Pavlovna's view about Russia's role in Europe?"
```

## Run the Ask HTTP Service

```bash
uvicorn server:app --host 0.0.0.0 --port 8010
```

Example requests:

```bash
curl http://localhost:8010/health
```

```bash
curl -X POST http://localhost:8010/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Anna Pavlovna'\''s view about Russia'\''s role in Europe?"}'
```
