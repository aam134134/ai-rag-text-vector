FROM python:3.12-slim

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

ARG PROJECT_NAME=ai-rag-text-vector-ask
ARG PROJECT_DIR=/app/home/${PROJECT_NAME}

RUN mkdir -p ${PROJECT_DIR}

WORKDIR ${PROJECT_DIR}
COPY ask_vector_data.py .
COPY query_vector_data.py .
COPY server.py .
COPY static ./static

EXPOSE 8010

ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8010"]
