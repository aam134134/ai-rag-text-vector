FROM python:3.12-slim

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ARG PROJECT_NAME=ai-rag-text-vector-text-ingest
ARG PROJECT_DIR=/app/home/${PROJECT_NAME}
ARG APP_DATA_HOME=/app/data/ingest

RUN mkdir -p ${PROJECT_DIR} 
RUN mkdir -p ${APP_DATA_HOME}

WORKDIR ${PROJECT_DIR}
COPY file_watcher.py .

ENTRYPOINT ["python", "-u","file_watcher.py"]
