FROM python:3.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Download libgomp that is essential for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN apt-get install -y unzip

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the project into the image
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Copy wordnet corpus to NLTK corpus folder
COPY resource/wordnet.zip /root/nltk_data/corpora/
RUN unzip /root/nltk_data/corpora/wordnet.zip

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV MLFLOW_ARTIFACT_ROOT=file:///mlflow/mlruns

EXPOSE 50

CMD uv run main.py && uv run mlflow server --host 0.0.0.0 --port 50 --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///mlflow/mlruns
