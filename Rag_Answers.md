# RAG Pipeline Interview Questions

Based on the arXiv Paper Curator system: arXiv API → Docling → PostgreSQL → OpenSearch → Ollama → FastAPI → LangFuse → Airflow → Gradio

---

## 1. RAG Architecture & Concepts

**Fundamentals**
1. What is Retrieval-Augmented Generation (RAG) and how does it differ from a pure LLM approach? What problems does it solve?
2. Walk me through the end-to-end flow of a RAG query — from user question to final answer.
3. What is the difference between BM25 (keyword) search and vector (semantic) search? When would you prefer one over the other?
4. What is hybrid search? How does Reciprocal Rank Fusion (RRF) combine BM25 and vector scores?
5. What is a chunk? Why do we chunk documents rather than indexing the full text? What are the trade-offs of chunk size?
6. What is an embedding? Why must the embedding model used at *query time* match the one used at *indexing time*?
7. What causes a "dimension mismatch" error in vector search, and how do you fix it?
8. What is cosine similarity and why is it commonly used for semantic search over dot product or L2 distance?

**Advanced**
9. What is the difference between `retrieval.query` and `retrieval.passage` task types in embedding models, and why does it matter?
10. How would you evaluate retrieval quality in a RAG system? What metrics would you use (MRR, NDCG, Recall@K)?
11. What is "semantic drift" in RAG and how do you mitigate it?
12. How does a guardrail work in an agentic RAG system? What does "query scope validation" mean in practice?
13. What is query rewriting in an agentic pipeline? When would you trigger a re-retrieval?

---

## 2. OpenSearch

**Fundamentals**
1. What is the difference between OpenSearch and Elasticsearch? Why might you choose OpenSearch?
2. What is an index mapping? Why do you define it explicitly rather than letting OpenSearch auto-map?
3. What is a `knn_vector` field? What parameters matter most when defining one (`dimension`, `engine`, `space_type`)?
4. What is HNSW and why is it used for approximate nearest-neighbour search?
5. What does `ef_construction` and `m` control in HNSW index parameters?
6. What is the difference between `filter` and `must` in a bool query?
7. What is a search pipeline in OpenSearch? How does the RRF pipeline work?

**Practical / Debugging**
8. We got `Query vector has invalid dimension: 1024. Dimension should be: 768` — what caused this and how did you fix it?
9. After a Docker volume wipe, what steps do you need to take to restore the OpenSearch index?
10. What is `refresh=True` in a bulk index call and when would you use it vs. leaving it off?
11. How would you check how many documents are in an index without using the UI?
12. What is the `_count` API vs `_stats` API in OpenSearch? When would you use each?
13. What is `cosinesimil` vs `l2` vs `innerproduct` as `space_type` values? Which is best for normalized embeddings?

---

## 3. Embeddings & Ollama

**Fundamentals**
1. What is the difference between a generative model and an embedding model?
2. What does it mean for an embedding to be 768-dimensional vs 1024-dimensional?
3. Why did we switch from Jina cloud API (1024-dim) to a local Ollama model (768-dim)? What were the trade-offs?
4. What is `jina-embeddings-v2-base-en`? What kind of tasks is it designed for?
5. What is Ollama? How does it serve models locally and what is its API interface?
6. What is the difference between `/api/embed` and `/api/embeddings` endpoints in Ollama, and why does it matter?
7. What is the difference between `embed_query` and `embed_passages`? Why do some models use different task types for each?

**Practical**
8. How do you batch embedding calls efficiently? What are the trade-offs of batch size?
9. We needed `OllamaEmbeddingsClient` to be a drop-in for `JinaEmbeddingsClient`. What interface methods did you need to match?
10. Why must `embed_query` and `embed_passages` be `async def` in FastAPI? What happens if they're sync?
11. How would you benchmark embedding throughput for a given model and batch size?
12. What is `qwen2.5:3b-instruct`? What does `3b` mean, and what is the difference between the base model and the instruct-tuned version?
13. If you wanted to use a larger, more capable model but had latency constraints, what strategies would you use?

---

## 4. FastAPI & API Design

**Fundamentals**
1. What is dependency injection in FastAPI? How does `Depends()` work?
2. What is `lru_cache` and why is it used on `get_settings()`? What problems can it cause in testing or during config changes?
3. What is a Pydantic `BaseModel`? How does it handle validation and serialization?
4. What is `Annotated` in Python type hints, and how does FastAPI use it for dependency injection?
5. What is a `lifespan` context manager in FastAPI? What goes in startup vs. teardown?
6. What is `StreamingResponse`? How do Server-Sent Events (SSE) work with it?
7. What is the difference between `app.state` and a regular module-level global in a FastAPI app?

**Practical / Debugging**
8. We saw `'LangfuseTracer' object has no attribute 'trace_rag_request'` — what type of error is this and how do you debug it systematically?
9. How does `openapi.json` help debug missing or misconfigured endpoints?
10. A `404 Not Found` on an endpoint that exists in code — what are the likely causes?
11. We had `make_agentic_rag_service() got unexpected keyword argument 'model'` — what caused this and how was it fixed?
12. What is the difference between `docker compose stop`, `docker compose down`, and `docker compose down -v`?
13. Why does editing `.env` not immediately affect a running Docker container? What steps are needed?

---

## 5. LangFuse Observability

**Fundamentals**
1. What is LLM observability and why is it important for production RAG systems?
2. What is a trace in LangFuse? How does it differ from a span and a generation observation?
3. What is the difference between LangFuse v2 and v3 Python SDK? How does the tracing API differ?
4. What does "OpenTelemetry-based" mean for LangFuse v3? How does context propagation work?
5. What is `flush_at` and `flush_interval` in LangFuse? Why do you flush before shutdown?
6. What is a Prisma migration? What happened when we mixed LangFuse v2 and v3 database schemas?
7. What is the difference between `start_as_current_span` and `start_span` in the v3 SDK?

**Practical / Debugging**
8. Traces appeared in the UI with `output: null` — what caused this and how would you debug it?
9. `LANGFUSE__HOST=http://localhost:3000` inside a Docker container — why is this wrong, and what should it be?
10. What is the difference between `LANGFUSE_HOST` (single underscore) and `LANGFUSE__HOST` (double underscore) in pydantic-settings? Which one gets read?
11. How would you verify that LangFuse is actually receiving traces vs. silently dropping them?
12. How does the LangFuse `CallbackHandler` integrate with LangChain/LangGraph, and how does it differ from manual span creation?
13. What is the `score_current_trace` / `create_score` API used for in LangFuse?

---

## 6. LangGraph & Agentic RAG

**Fundamentals**
1. What is LangGraph? How does it differ from a simple chain (LangChain)?
2. What is a node in a LangGraph graph? What is an edge? What is conditional routing?
3. What is `GraphConfig` in a LangGraph service? What parameters would you typically configure?
4. What is a guardrail node in an agentic RAG system? What does "query scope validation" mean?
5. What is query rewriting and why might an agent perform multiple retrieval attempts?
6. What is the difference between `AgenticRAGService` and a simple `rag_ask()` function? When would you prefer one over the other?
7. What does `retrieval_attempts: 0` signify in an agentic RAG response?

**Practical**
8. How do you pass per-request parameters (like `model`) to a LangGraph that was constructed once at dependency injection time?
9. What is the `reasoning_steps` field in an agentic response? How is it populated?
10. How would you add a document grading/relevance filtering node to the existing agentic pipeline?
11. What are the latency trade-offs of an agentic vs. non-agentic RAG pipeline?

---

## 7. PostgreSQL & SQLAlchemy

**Fundamentals**
1. What is SQLAlchemy? What is the difference between the Core and ORM layers?
2. What is `get_session()` as a context manager? Why is session lifecycle management important?
3. What is an upsert? How does `paper_repo.upsert(paper_create)` work?
4. What is `lru_cache` on `make_database()`? What are the risks and benefits?
5. What is connection pooling (`pool_size`, `max_overflow`)? Why does it matter for a web API?
6. What is the difference between `filter()` and `filter_by()` in SQLAlchemy ORM queries?
7. What does `Paper.raw_text.isnot(None)` do in a SQLAlchemy query?

**Practical / Debugging**
8. We repeatedly hit `password authentication failed for user "rag_user"` — what were the causes and how was each fixed?
9. Why does setting `POSTGRES_DATABASE_URL` as an env var need to happen *before* any `from src.config import get_settings` import in a Jupyter notebook?
10. The Postgres container exposes port 5433 on the host but 5432 internally — explain how Docker port mapping works and when you use each.

---

## 8. Docker & Docker Compose

**Fundamentals**
1. What is the difference between a Docker image and a container?
2. What is a Docker volume? What is the difference between a named volume and a bind mount?
3. What does `docker compose down -v` do differently from `docker compose down`?
4. Why does `docker compose stop` not free a volume? What command actually removes the container reference?
5. What is a `healthcheck` in `compose.yml`? How do `depends_on` conditions use it?
6. What is the difference between `container_name` and the compose service name? Which does `docker compose up -d <name>` use?
7. What is `--force-recreate` vs `--build` vs `--no-cache` in Docker Compose?

**Practical / Debugging**
8. Environment variable precedence: if `compose.yml` has an explicit `environment:` block *and* an `env_file:` directive both setting the same key — which wins?
9. A container shows as `unhealthy` but is running — how do you diagnose it?
10. We saw `[Errno -3] Temporary failure in name resolution` from inside a container — what caused this and how was it fixed?
11. What is `rag-network` as a bridge network? Why do services need to be on the same network to talk to each other by name?
12. `LANGFUSE__HOST=http://localhost:3000` vs `http://langfuse-web:3000` — explain why one works and one doesn't from inside a container.

---

## 9. Airflow

**Fundamentals**
1. What is Apache Airflow? What problem does it solve?
2. What is a DAG in Airflow? What is a task? What is an operator?
3. What does "paused" mean for a DAG in the Airflow UI?
4. What is `airflow dags list-import-errors` and when would you use it?
5. What is the difference between `airflow dags backfill` and manually triggering a DAG?
6. Why was `docling` not available in the Airflow container even though it worked in the API container?

---

## 10. Python & Software Engineering

**Fundamentals**
1. What is `@lru_cache` and what are its risks when used on settings/config functions?
2. What is `async def` vs `def` in Python? When is `await` required?
3. Why does calling `asyncio.run()` inside a Jupyter notebook throw `RuntimeError: asyncio.run() cannot be called from a running event loop`?
4. What is `@contextmanager` and `yield` in Python? How does it enable `with` block semantics?
5. What is the difference between `Annotated[T, Depends(fn)]` and just `T = Depends(fn)` in FastAPI?
6. What is pydantic-settings and how does `env_nested_delimiter="__"` work? (e.g. `LANGFUSE__HOST` → `settings.langfuse.host`)
7. What is `inspect.signature()` useful for when debugging SDK/API mismatches?

**Practical**
8. Describe a strategy for managing notebook kernel state across long debugging sessions to avoid `NameError: name 'X' is not defined`.
9. How does `requests.post(..., stream=True)` work and how do you consume an SSE stream line-by-line?
10. What are `__all__` and `from . import X` in `__init__.py`? How does Python package import resolution work?

---

## 11. System Design

1. Design a RAG system that can scale to 1 million documents. What changes would you make to this architecture?
2. How would you add multi-tenancy (per-user isolated data) to this system?
3. The system currently has 3/33 papers indexed (those with `raw_text`). Design a pipeline to parse and index all 33 at once, with error handling and retries.
4. How would you implement semantic caching (not just exact-match Redis caching) to speed up repeated similar queries?
5. How would you A/B test two different embedding models or chunking strategies in production?
6. What monitoring would you add to detect when retrieval quality degrades over time?
7. How would you handle a paper that's too large for a single context window when generating an answer?
8. Design the ingestion pipeline to automatically pick up new arXiv papers daily and index them without re-indexing existing ones.
9. If the Ollama server at `10.2.2.183` goes down, how would you design the system to fail gracefully?
10. What would you change if you needed the system to support real-time streaming responses to 1000 concurrent users?

---

## Quick Reference — Core Configs in This System

```
OpenSearch index:  arxiv-papers-chunks
Embedding model:   jina/jina-embeddings-v2-base-en (768-dim, on 10.2.2.183:11434)
LLM model:         qwen2.5:3b-instruct (on 10.2.2.183:11434)
Vector space:      cosinesimil (HNSW via nmslib)
Chunk size:        600 words, 100-word overlap
Postgres port:     5433 (host) → 5432 (container)
LangFuse UI:       http://localhost:3001
```
