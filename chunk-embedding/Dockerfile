FROM python:3.12-slim

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ARG PROJECT_NAME=ai-rag-text-vector-chunk-embedding
ARG PROJECT_DIR=/app/home/${PROJECT_NAME}
ARG APP_DATA_HOME=/app/data/chunking

RUN mkdir -p ${PROJECT_DIR} 
RUN mkdir -p ${APP_DATA_HOME}

WORKDIR ${PROJECT_DIR}
COPY embed_chunks.py .

ENTRYPOINT ["python", "-u","embed_chunks.py"]
