FROM python:3.12-slim

ARG PROJECT_NAME=ai-rag-text-vector-text-ingest
ARG USER_HOME=/home/appuser
ARG PROJECT_DIR=${USER_HOME}/${PROJECT_NAME}

RUN pip install uv==0.10.7

RUN useradd -m -u 1000 appuser
USER appuser

RUN mkdir -p ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

RUN uv init
RUN uv venv
RUN uv add unstructured==0.21.5
RUN echo "source \"${PROJECT_DIR}/.venv/bin/activate\"" >> ~/.bashrc

COPY file_watcher.py ${PROJECT_DIR}/file_watcher.py

ENTRYPOINT ["./.venv/bin/python", "-u","file_watcher.py"]
