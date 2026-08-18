FROM python:3.11-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app /app

RUN pip install --no-cache-dir pytest

CMD ["bash"]
