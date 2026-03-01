FROM python:3.14.3-bookworm

ARG PROJECT_NAME=set_a_name
ARG USER_HOME=/home/set_a_folder
ARG PROJECT_DIR=${USER_HOME}/${PROJECT_NAME}

RUN pip install uv==0.10.7

RUN useradd -m -u 1000 appuser
USER appuser

RUN mkdir -p ${PROJECT_DIR}
RUN mkdir -p ${USER_HOME}/inbox
WORKDIR ${PROJECT_DIR}

RUN uv init
RUN uv venv
RUN echo "source \"${PROJECT_DIR}/.venv/bin/activate\"" >> ~/.bashrc

RUN uv add unstructured==0.21.5

CMD ["python"]

