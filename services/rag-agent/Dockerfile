# Built from the repo root so the repo's markdown docs can be indexed.
FROM python:3.11-slim

WORKDIR /app

COPY services/rag-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/rag-agent/agent agent
COPY . /docs

# Index docs at build time (also caches the embedding model in the image)
ENV CHROMA_DIR=/app/chroma DOCS_DIRS=/docs
RUN python -m agent.store

EXPOSE 8001

CMD ["uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8001"]
