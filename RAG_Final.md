# RAG Pipeline Interview Q&A

Based on the arXiv Paper Curator system: arXiv API → Docling → PostgreSQL → OpenSearch → Ollama → FastAPI → LangFuse → Airflow → Gradio

---

## 1. RAG Architecture & Concepts

**Q1. What is RAG and how does it differ from a pure LLM approach?**

RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM. Instead of relying solely on knowledge baked into model weights during training, RAG first searches a document store for relevant content, then passes that content as context to the LLM to generate an answer grounded in real data.

Problems it solves: 

Hallucination — the LLM answers from retrieved facts, not invented ones
Knowledge cutoff — can query fresh documents the model was never trained on
Domain specificity — your arXiv papers aren't in any public LLM's training data
Auditability — you can cite sources (the chunks that were retrieved)
Cost — cheaper than fine-tuning a model on your corpus
In our system: user asks a question → OpenSearch retrieves relevant chunks from indexed arXiv papers → qwen2.5:3b-instruct generates an answer grounded in those chunks.

---

**Q2. Walk me through the end-to-end RAG query flow.**

1. User submits query ("How does diffusion-based code repair work?")
2. FastAPI `/api/v1/ask` receives the request (validated by Pydantic AskRequest)
3. OllamaEmbeddingsClient embeds the query into a 768-dim vector
4. OpenSearch `search_unified()` runs hybrid search: BM25 keyword match + kNN vector search, combined via RRF pipeline → returns top-k chunks
5. Retrieved chunks are formatted into a prompt by RAGPromptBuilder
6. OllamaClient sends the prompt to qwen2.5:3b-instruct, which generates an answer
7. LangFuse records the full trace (embedding → retrieval → prompt → generation)
8. Redis caches the response for exact-match future queries
9. AskResponse (answer + sources + chunks_used + search_mode) returned to user


User question
    ↓
embed_query()          # OllamaEmbeddingsClient → 768-dim vector via jina-embeddings-v2-base-en
    ↓
search_unified()       # OpenSearch: BM25 + kNN vector search → RRF fusion → top-k chunks
    ↓
RAGPromptBuilder       # Formats chunks + question into a structured prompt
    ↓
ollama_client.generate_rag_answer()   # POST /api/generate → qwen2.5:3b-instruct
    ↓
AskResponse            # {answer, sources, chunks_used, search_mode}
    ↓
LangFuse trace         # Records embedding span, search span, generation observation
    ↓
Redis cache            # Store response for exact-match future queries
---

**Q3. BM25 vs vector search — differences and when to prefer each.**

BM25 (Best Match 25) is a keyword-based ranking algorithm that scores documents by term frequency and inverse document frequency. It's fast, interpretable, and works well when the exact keyword matters (e.g. searching for "arxiv_id:2508.11110").

Vector search embeds both query and documents into a continuous vector space and ranks by cosine similarity. It captures semantic meaning — "code fixing" and "bug repair" are similar even with no shared words.

Prefer BM25 when: exact terminology matters, short queries, high-precision keyword lookups.
Prefer vector search when: paraphrased queries, semantic relationships matter, cross-lingual search.

In our system we use both combined (hybrid), because research papers use precise terminology (BM25 helps) but user queries may be phrased differently from paper language (vector helps).

In practice: hybrid search (both combined via RRF) almost always outperforms either alone, which is why we have search_chunks_hybrid as the default.

**Q4. What is hybrid search and how does RRF work?**

Hybrid search runs BM25 and vector search independently and merges the result lists. Reciprocal Rank Fusion (RRF) is the merging algorithm:

RRF_score(doc) = Σ 1 / (k + rank_in_each_list)
Where k is a constant (typically 60) that prevents very high ranks from dominating. A document ranked 1st in both BM25 and vector gets the highest combined score; a document ranked 1st in only one list gets a moderate score.

In our OpenSearch setup this is handled by a search pipeline (hybrid-rrf-pipeline) configured as rrf_pipeline_name in settings. The pipeline runs both queries and merges automatically — our code just calls search_unified(..., use_hybrid=True).

---

**Q5. What is a chunk and why chunk documents?**

A chunk is a fixed-size segment of a document's text. We chunk because:

LLM context windows are limited (can't pass a whole 80k-char paper)
Smaller units improve retrieval precision (retrieve the specific section, not the whole paper)
Embedding quality degrades on very long texts
Trade-offs by chunk size:

Smaller chunks	Larger chunks
Higher precision retrieval	More context per retrieved unit
May miss context that spans boundaries	May dilute relevance signal
Need more chunks to cover the paper	Fewer chunks, faster indexing
Better for fact lookup	Better for conceptual questions
In our system: 600-word chunks with 100-word overlap. Overlap prevents answers from being split across chunk boundaries.

**Q6. What is an embedding and why must query/index models match?**

An embedding is a dense vector that represents the semantic meaning of text in a high-dimensional space. Similar texts produce vectors that are close together (high cosine similarity).

Query and indexing models must match because:

Each model creates its own geometric space — vectors from different models are in different coordinate systems
A query embedded by Model A searching against documents embedded by Model B is like looking for matching GPS coordinates on a different planet's map
Even same-architecture models with different training produce incompatible spaces

If you use a different model at query time, the vectors live in entirely different geometric spaces — "diffusion" at query time might map to the direction closest to "table formatting" in the index space, making similarity scores meaningless. Same model, same space.

---

**Q7. What causes a dimension mismatch error and how do you fix it?**

Error: `Query vector has invalid dimension: 1024. Dimension should be: 768`

Cause: the OpenSearch index was created with `knn_vector` field dimension 768 (matching our Ollama jina-v2 model), but the query embedding was generated by a different model producing 1024-dim vectors (Jina cloud API v3 model).

Fix:
1. Delete the existing index
2. Recreate with the correct dimension matching your actual embedding model
3. Re-index all documents using the same model
4. Ensure `OPENSEARCH__VECTOR_DIMENSION=768` in config so setup_indices() creates the right mapping

In our case, the root cause was that `JinaEmbeddingsClient` (cloud, 1024-dim) was still used for query embedding even after we'd already indexed with `OllamaEmbeddingsClient` (local, 768-dim).

---

**Q8. Why is cosine similarity preferred over dot product or L2 for semantic search?**

Cosine similarity measures the angle between two vectors, ignoring magnitude. This means the length of the embedding vector doesn't affect the similarity score — only the direction matters. This is ideal for text embeddings because a longer document doesn't inherently mean higher relevance.

L2 (Euclidean distance) is sensitive to magnitude — a 768-dim vector with larger values would appear "further" from everything. Inner product (dot product) is equivalent to cosine similarity only when vectors are normalized to unit length (which most embedding models do, making inner product a valid and faster alternative when vectors are normalized).

Our OpenSearch index uses `cosinesimil` as `space_type` in the HNSW configuration.

Why prefer it:

Magnitude-invariant — a short text and a long text about the same topic have similar cosine similarity even if their raw vectors have very different magnitudes
Standard for semantic search — most embedding models are trained with cosine similarity as the objective
Dot product rewards high-magnitude vectors which can skew results
L2 (Euclidean distance) is magnitude-sensitive in the same way dot product is
In our OpenSearch config: "space_type": "cosinesimil" in the HNSW method parameters.


---

**Q9. retrieval.query vs retrieval.passage task types.**

Some embedding models (including Jina v3) are asymmetric — they embed queries and passages differently to optimize retrieval. `retrieval.query` applies a transformation optimized for short search queries (emphasizing discriminative features). `retrieval.passage` is optimized for longer document chunks (emphasizing content representation).
Why it matters: if you embed documents with `retrieval.passage` but query with `retrieval.passage` too (or vice versa), retrieval quality degrades. The asymmetry is intentional — queries and passages have different linguistic properties.
In our system `jina-embeddings-v2-base-en` via Ollama doesn't expose task types (it's a symmetric model), so both query and passage use the same embedding — which simplified our OllamaEmbeddingsClient interface.


Many modern embedding models (including Jina v3) use asymmetric embeddings — they produce different-shaped representations depending on whether the input is a short query or a longer passage:
retrieval.query: optimised for short, natural-language questions. Produces vectors tuned to find documents that answer the question.
retrieval.passage: optimised for indexing longer document chunks. Produces vectors tuned to be retrieved by query vectors.
This asymmetry improves retrieval accuracy because the model learns that "how does X work?" should be close to "X works by doing Y and Z" in embedding space, even though they're structurally different.

In JinaEmbeddingsClient: embed_query() sends task: "retrieval.query", embed_passages() sends task: "retrieval.passage". Our OllamaEmbeddingsClient uses jina-embeddings-v2-base-en (v2, not v3) which doesn't have this task separation — it uses symmetric embeddings. This is a quality trade-off we accepted to avoid the cloud API 403 issue.


---

**Q10. How would you evaluate retrieval quality?**

Key metrics:
- **Recall@K**: of all truly relevant documents, what fraction appeared in the top K results? Measures coverage.
- **Precision@K**: of the top K results returned, what fraction are actually relevant? Measures accuracy.
- **MRR (Mean Reciprocal Rank)**: average of 1/rank of the first relevant result. Rewards finding the right answer early.
- **NDCG (Normalized Discounted Cumulative Gain)**: accounts for graded relevance (very relevant > somewhat relevant > not relevant) and position.

Practical approach for our system: 
create a golden evaluation set of (question, relevant_arxiv_ids) pairs. 
Run retrieval for each question. Score against ground truth. 
LangFuse can store these evaluations as scores attached to traces.

Create a golden test set: 20–30 questions with known correct arxiv_ids that should be retrieved
Run retrieval and measure Recall@3 and MRR
Compare BM25-only vs vector-only vs hybrid — hybrid almost always wins
Use LangFuse scores to capture user feedback on answer quality as a proxy for retrieval quality


---

**Q11. What is semantic drift in RAG and how do you mitigate it?**

Semantic drift happens when the retrieved documents are topically related but don't actually answer the question — the retrieval "drifts" toward loosely related content.

Example: query "what is the training loss of the Tabularis model?" retrieves chunks about loss functions in general (semantic match) rather than the specific paper's training details.

Mitigations:

Document grading node (in agentic RAG): after retrieval, have the LLM score each chunk for relevance to the specific question; discard low-score chunks
Metadata filtering: filter by categories, arxiv_id, or published_date to narrow the search space
Query rewriting: if first retrieval drifts, rewrite the query more specifically and retry
Smaller, more precise chunks: large chunks dilute the relevance signal
Hybrid search: BM25 anchors results to query keywords, preventing pure semantic drift
**Q12. How does a guardrail work in an agentic RAG system?**

A guardrail is a validation node in the LangGraph pipeline that runs before retrieval. It scores the incoming query for relevance to the system's domain, and rejects queries that fall below a threshold.

In our system:

LangGraph node calls the LLM (qwen2.5:3b-instruct) to rate the query from 0–100 for relevance to CS/AI/ML research
Score < 60: rejected — agent sets retrieval_attempts: 0 and returns a polite out-of-scope message
Score ≥ 60: proceeds to retrieval
"What is a dog?" scored 50 → rejected immediately. "How does diffusion-based code repair work?" would score ~90 → proceeds.

This prevents the system from hallucinating answers to off-topic questions and saves the latency cost of a full retrieval + generation cycle.

---

**Q13. What is query rewriting and when would you trigger re-retrieval?**

Query rewriting is the process of transforming the original user query into a better search query when the initial retrieval doesn't find relevant documents.

Triggers:

Retrieved chunks have low relevance scores (below a threshold)
The LLM grades the retrieved chunks as insufficient to answer the question
Maximum retrieval attempts not yet reached (e.g., max_attempts: 2)
How it works in LangGraph:

First retrieval with original query → chunks graded as insufficient
Rewrite node: LLM generates a more specific or differently-phrased query
Second retrieval with rewritten query → grade again
If still insufficient: return "no relevant information found"
Example: "Tell me about code repair stuff" → rewrites to "diffusion model code repair generation arXiv" → retrieves paper 2508.11110v1.

## 2. OpenSearch

**Q1. OpenSearch vs Elasticsearch.**

OpenSearch is a community-driven, Apache 2.0 licensed fork of Elasticsearch 7.10, created in 2021 after Elastic changed its licensing. Key differences:

License: OpenSearch is fully open-source (Apache 2.0); Elasticsearch has dual licensing (SSPL/ELv2) which restricts cloud providers from offering it as a managed service
k-NN plugin: OpenSearch has a native k-NN plugin with HNSW, IVF, and FAISS support — this is what we use for vector search
ML Commons: OpenSearch has built-in ML inference; Elasticsearch has similar but via Elastic ML
Managed services: AWS OpenSearch Service, not available for Elasticsearch
We chose OpenSearch because AWS offers it as a managed service (useful for Azure deployment migration), it has a strong k-NN plugin for vector search, and it's free of licensing restrictions.

---

**Q2. What is an index mapping and why define it explicitly?**

An index mapping is the schema for an OpenSearch index — it defines field names, data types, and indexing/analysis settings for each field.

Explicitly defining it matters because:

Auto-mapping guesses types — a field that looks like a string gets mapped as text with full-text analysis, but you might need keyword for exact-match filtering or aggregations
knn_vector fields can't be auto-mapped — OpenSearch doesn't know to create a vector field; you must specify "type": "knn_vector" with dimension and method
Strict mode ("dynamic": "strict") prevents accidental fields from being indexed — cleaner schema
Analyzers — our mapping uses text_analyzer for chunk_text (enables full-text BM25 search) and keyword for arxiv_id (exact match filtering)

---

**Q3. knn_vector field and its key parameters.**

knn_vector is a special OpenSearch field type for storing dense vectors and enabling approximate nearest-neighbour search.

```json
{
  "type": "knn_vector",
  "dimension": 768,
  "method": {
    "name": "hnsw",
    "engine": "nmslib",
    "space_type": "cosinesimil",
    "parameters": {
      "ef_construction": 512,
      "m": 16
    }
  }
}
```

- `dimension`: must exactly match your embedding model's output size
- `engine`: `nmslib` or `faiss` — both support HNSW; nmslib is default
- `space_type`: distance metric (`cosinesimil`, `l2`, `innerproduct`)
- `name`: algorithm — `hnsw` (Hierarchical Navigable Small World) is standard
- `ef_construction`: size of the candidate list during index build — higher = better accuracy, slower indexing
- `m`: number of bi-directional links per node — higher = better recall, more memory

---

**Q4. What is HNSW and why use it for ANN search?**

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbour (ANN) algorithm.
It builds a multi-layer graph where the top layers are sparse (fast navigation) and lower layers are dense (precise search).
To find the nearest neighbour of a query vector, it starts at the top layer, greedily navigates toward the query, 
then descends to more precise layers.

Why ANN instead of exact search? Exact nearest-neighbour search in 768 dimensions requires comparing the query to every stored vector — O(n) time. With 81 chunks that's fine, but at 1M+ documents it's too slow. HNSW achieves sub-linear search time with controllable accuracy trade-off (~10ms at 99% recall for millions of vectors).

Each node represents a document vector
Edges connect nearby vectors
Higher layers are sparser (faster traversal); lower layers are denser (more accurate)
At search time, it starts at a random node in the top layer, navigates greedily toward the query vector, then descends through layers for increasingly fine-grained search.


---

**Q5. ef_construction and m parameters.**

m (max connections per node): controls the connectivity of the graph. Higher m = more edges = better recall but more memory. Default 16 is good for most use cases; increase to 32–64 for high-recall requirements.

ef_construction (size of the dynamic candidate list during index build): controls the quality-vs-speed trade-off at indexing time. Higher = better index quality (finds better neighbours during construction), slower indexing. Our value of 512 is on the high end — appropriate for a small corpus where index quality matters more than build speed.

There's also ef_search (set at query time) which controls the same trade-off during search. We don't set it explicitly, so it uses the default.

**Q6. filter vs must in a bool query.**
In OpenSearch's bool query:

must: the document must match this clause AND its score contributes to relevance ranking
filter: the document must match this clause BUT the score is NOT affected (binary pass/fail)
Use filter for: category filters, date ranges, exact-field matches — anything that narrows the result set without changing relevance ordering. Filters are also cached automatically by OpenSearch, making them faster.

Use must for: the main search clause where relevance scoring matters.

In our search_chunks_vector method:

"filter": [{"terms": {"categories": categories}}]
Category filtering uses filter (not must) so it doesn't interfere with the cosine similarity score.

In our system: `search_chunks_vector` optionally applies a `filter` on the `categories` field — it narrows results to specific arXiv categories without affecting the kNN similarity score.

---

**Q7. What is a search pipeline in OpenSearch and how does RRF work?**

A search pipeline is a configurable chain of processors that modify search requests or results. Our `hybrid-rrf-pipeline` uses the `normalization-processor` with `technique: rrf` to merge results from multiple search types.

RRF formula: `score(doc) = Σ 1 / (60 + rank_i)` across all result lists. A document ranked #1 in both BM25 and vector results gets `1/61 + 1/61 ≈ 0.033`. A document ranked #1 in BM25 only gets `1/61 ≈ 0.016`. The constant 60 prevents a very high rank from dominating.

Receives results from both BM25 and k-NN vector searches
Ranks documents in each list (BM25 rank, vector rank)
Applies RRF formula: score = Σ 1 / (60 + rank) for each list
Sorts by combined RRF score
Returns top-k merged results
It's configured in setup_indices() via _create_rrf_pipeline() and referenced by name in hybrid search queries via the search_pipeline parameter.

**Q8. Dimension mismatch — what caused it and how was it fixed?**

Root cause: We had two different embedding clients in use simultaneously.
- **Index time**: `OllamaEmbeddingsClient` (jina-v2 via Ollama, 768-dim) correctly indexed 77 chunks
- **Query time**: `JinaEmbeddingsClient` (Jina cloud API v3, 1024-dim) tried to search — but the index was built for 768-dim vectors

OpenSearch's kNN engine cannot compare a 1024-dim query vector against 768-dim stored vectors, so it threw the error.

Fix: deleted the index, recreated with `dimension: 768`, re-indexed all chunks, and ensured the production API also uses `OllamaEmbeddingsClient` (via `dependencies.py` swap and `OLLAMA_HOST` pointing to `10.2.2.183`).

# 1. Check current dimension
mapping = opensearch_client.client.indices.get_mapping(index=index_name)
current_dim = mapping[index_name]["mappings"]["properties"]["embedding"]["dimension"]

# 2. Delete old index (cannot update knn_vector dimension in-place)
opensearch_client.client.indices.delete(index=index_name)

# 3. Recreate with 768-dim
new_mapping = copy.deepcopy(ARXIV_PAPERS_CHUNKS_MAPPING)
new_mapping["mappings"]["properties"]["embedding"]["dimension"] = 768
opensearch_client.client.indices.create(index=index_name, body=new_mapping)

# 4. Re-index all documents with new embedding model
await service.index_papers_batch(papers_data, replace_existing=True)
Also update OPENSEARCH__VECTOR_DIMENSION=768 in .env so setup_indices() creates the right schema on future container restarts.


---

**Q9. Restoring OpenSearch index after a volume wipe.**

1. Confirm the index no longer exists: `curl http://localhost:9200/_cat/indices`
2. Re-create the index with the correct mapping (dimension, analyzers, settings)
3. Re-run the indexing pipeline: load parsed papers from PostgreSQL, chunk each `raw_text`, generate embeddings via `OllamaEmbeddingsClient`, bulk index via `HybridIndexingService.index_papers_batch()`
4. Verify: `curl http://localhost:9200/arxiv-papers-chunks/_count` should show expected doc count
5. Run a test vector search to confirm embeddings are searchable

In our case this took ~5 minutes for 77 chunks (dominated by Ollama embedding time).


A volume wipe deletes all persisted data. Steps to restore:

Recreate the index with correct mapping (setup_indices() runs on API startup, but verify vector_dimension=768)
Ensure papers exist in PostgreSQL (if Postgres volume was also wiped, re-run the Week 2 ingestion pipeline)
Re-parse PDFs via Docling for papers where raw_text is needed (only 3 in our case — 2508.11110v1, 2508.11112v1, 2508.11121v1)
Re-run index_papers_batch() with OllamaEmbeddingsClient to embed and index all chunks
Verify: curl http://localhost:9200/arxiv-papers-chunks/_count should show 77–81 documents
Prevention: use bind-mount volumes for production data, or take OpenSearch snapshots regularly.

---

**Q10. refresh=True in bulk indexing.**

`refresh=True` forces OpenSearch to make the indexed documents immediately searchable after the bulk operation returns. Without it, documents are only searchable after the next automatic refresh cycle (default: every 1 second).

Use `refresh=True` when: you need to verify the index immediately (e.g. test assertions right after indexing). 
Avoid in production high-throughput scenarios because forcing a refresh on every bulk call is expensive — it requires a Lucene segment merge.

Don't use it in production high-throughput indexing: forcing a refresh after every bulk is expensive — it triggers a segment merge and disk flush. For batch ingestion jobs (like our Airflow DAG), omit refresh=True and let the natural 1-second cycle handle it.

**Q11. Checking document count without the UI.**

```bash
# Count documents in an index
curl http://localhost:9200/arxiv-papers-chunks/_count

# Response: {"count": 81, "_shards": {...}}

# List all indices with doc counts
curl http://localhost:9200/_cat/indices?v
```

Our system showed 81 documents across 3 papers (77 from initial indexing + 4 from a 4th cached PDF re-index).

---
# Via Python
stats = opensearch_client.client.count(index="arxiv-papers-chunks")
print(stats["count"])

# Or via get_index_stats() (our wrapper)
stats = opensearch_client.get_i
**Q12. _count API vs _stats API.**

`_count` returns just the document count (optionally filtered by a query). Fast, lightweight, used for existence checks.

`_stats` returns detailed index statistics: document count, deleted docs, store size in bytes, indexing/search operations, merge counts, cache stats. Used for monitoring index health, storage usage, and performance profiling.

In our notebooks we used `get_index_stats()` (wraps `_stats`) to show document count + size in bytes.

---

**Q13. cosinesimil vs l2 vs innerproduct.**

- `cosinesimil`: measures angle between vectors, ignoring magnitude. Best for text embeddings where length shouldn't affect relevance. Higher score = more similar. Our choice.
- `l2`: Euclidean distance — straight-line distance between points. Sensitive to vector magnitude. Lower score = more similar (distance, not similarity). Good for image embeddings or when magnitude carries meaning.
- `innerproduct`: dot product. Equivalent to cosine similarity when vectors are unit-normalized (which most embedding models do). Faster to compute than cosine but numerically equivalent for normalized embeddings.

For our normalized 768-dim text embeddings, `cosinesimil` and `innerproduct` would give identical rankings, but `cosinesimil` is more intuitive and explicit.

Rule of thumb: if your embedding model documentation says "cosine similarity", use cosinesimil. Jina-embeddings-v2 is trained for cosine similarity.

---

## 3. Embeddings & Ollama

**Q1. Generative model vs embedding model.**

A generative model (e.g. qwen2.5:3b-instruct) produces new text token-by-token given an input prompt. It's decoder-based, autoregressive, and can write answers, stories, code.

An embedding model (e.g. jina-embeddings-v2-base-en) maps any input text to a fixed-size dense vector. It produces no text output — only a numeric representation. It's typically encoder-based (like BERT). The vector captures semantic meaning for tasks like similarity search, clustering, classification.

In our system: `jina-embeddings-v2-base-en` embeds paper chunks and queries → OpenSearch similarity. `qwen2.5:3b-instruct` generates the actual answer text.
You can't use a generative model for embedding (well, you can extract internal hidden states, but that's not standard) and you can't use an embedding model for generation.

---

**Q2. 768-dim vs 1024-dim embeddings.**

The dimension is the length of the embedding vector — the number of float values used to represent a piece of text. Higher dimension = more expressive (can represent more subtle semantic distinctions) but requires more storage, more RAM, and slower similarity computation.

768-dim: BERT-style base models, jina-v2-base, many standard models. Good balance of quality and efficiency.
1024-dim: larger models like jina-v3, OpenAI ada-002 (1536-dim). Higher quality, higher cost.

The critical constraint: once you choose a dimension and build an index, you cannot mix. All vectors in one index must have the same dimension.

The number of dimensions is the length of the output vector. A 768-dim embedding is a list of 768 floating-point numbers. More dimensions generally:

Capture more nuanced semantic distinctions
Require more memory per vector (768 × 4 bytes = 3KB per chunk)
Require more computation for similarity search
Don't always mean better quality — it depends on training
jina-embeddings-v2-base-en: 768-dim, 136M parameters, optimised for retrieval tasks. jina-embeddings-v3 (cloud): 1024-dim, larger model, generally higher quality but required cloud API (which 403'd on our free tier).

Trade-off we made: lower dimension + local hosting = no API cost, no 403 errors, lower latency.
---

**Q3. Why switch from Jina cloud API to local Ollama?**

The Jina cloud API (jina-embeddings-v3, 1024-dim) kept returning `403 Forbidden` — the free-tier quota was exhausted after indexing 77 chunks during testing. This made query-time embedding impossible.

Trade-offs:
- **Local Ollama (jina-v2, 768-dim)**: free, no rate limits, private data stays local, works offline. Slower than cloud API (CPU inference). Requires a server with the model pulled.
- **Jina cloud (v3, 1024-dim)**: higher quality embeddings, faster inference, no GPU needed. But costs money, rate-limited, data leaves your network, requires API key management.
For our use case (private arXiv papers, development environment, server at 10.2.2.183 already running), local was the clear choice.


Trade-offs of switching:

Jina Cloud (v3, 1024-dim)	Local Ollama (v2-base-en, 768-dim)
Quality	Higher (larger model, asymmetric)	Good (smaller, symmetric)
Latency	Network round-trip	Local (~50ms/request)
Cost	Free tier quota, then paid	Zero (already have the GPU server)
Reliability	External dependency	Fully controlled
Dimension	1024	768
For a personal RAG project on a small corpus, local Ollama was the pragmatic choice. For production at scale, Jina v3 or a comparable high-quality model is worth paying for.

---

**Q4. What is jina-embeddings-v2-base-en?**

A 137M parameter English embedding model from Jina AI, based on a modified BERT architecture that supports 8192 token input (vs BERT's 512). It produces 768-dimensional embeddings. Designed for retrieval tasks — asymmetric retrieval, semantic similarity, clustering.

"v2-base" means: second version, base model size (vs large). The Ollama version (`jina/jina-embeddings-v2-base-en`) is a GGUF-quantized version (F16 in our case, per `ollama list` output showing `quantization_level: F16`).


Key characteristics:

English-only (the -en suffix)
8192 token max input (vs BERT's 512) — good for long paper sections
Symmetric embeddings (v2 doesn't have separate query/passage task types like v3)
Available as a GGUF model for Ollama under jina/jina-embeddings-v2-base-en
---

**Q5. What is Ollama and how does it serve models?**

Ollama is a local LLM serving tool that simplifies downloading, managing, and running open-source models. It wraps llama.cpp (the inference engine) with a simple REST API and CLI.

How it works:
- Models are stored in GGUF format (quantized for CPU/GPU efficiency)
- `ollama serve` starts an HTTP server on port 11434
- `/api/generate` for text generation (streaming or non-streaming)
- `/api/embed` for embedding generation
- `/api/tags` lists available models

In our system, Ollama runs on a dedicated server at 10.2.2.183 with a GPU, serving both the qwen2.5:3b-instruct chat model and jina-embeddings-v2 embedding model.


Ollama is an open-source tool for running LLMs and embedding models locally. It:

Downloads and manages GGUF-quantized model files
Exposes a REST API compatible with many LLM client libraries
Handles batching, GPU acceleration (CUDA/Metal), and memory management
Runs as a daemon (ollama serve) on the configured port (default 11434)
API interface (key endpoints):

GET  /api/tags         → list available models
POST /api/generate     → text generation (stream or non-stream)
POST /api/embed        → batch embedding (newer API)
POST /api/embeddings   → single embedding (older API)
GET  /api/version      → health check
In our system Ollama runs on the remote GPU server 10.2.2.183:11434 and serves both qwen2.5:3b-instruct (generation) and jina/jina-embeddings-v2-base-en (embeddings).


---

**Q6. /api/embed vs /api/embeddings in Ollama.**

`/api/embeddings` (older): takes `{"model": "...", "prompt": "..."}` — single string input, returns `{"embedding": [...]}`. Used in older Ollama versions.
`/api/embed` (newer): takes `{"model": "...", "input": ["...", "..."]}` — supports list input for batch embedding, returns `{"embeddings": [[...], [...]]}`. Our `OllamaEmbeddingsClient._embed()` uses `/api/embed` with list input for batching.
The distinction matters: the older endpoint doesn't batch, requiring N API calls for N texts. The newer endpoint allows N texts in one call — ~10x throughput improvement for large indexing jobs.
{"model": "jina/...", "prompt": "text"}
→ {"embedding": [0.1, 0.2, ...]}
/api/embed (newer, v0.1.31+): accepts "input" which can be a string or list of strings, returns "embeddings" as a list of lists. Supports batch processing natively.

{"model": "jina/...", "input": ["text1", "text2"]}
→ {"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}
We use /api/embed in OllamaEmbeddingsClient._embed() because it supports batching. When we first tried /api/embeddings with "prompt", we got Embedding dimension: 0 because the response key was different.

---

**Q7. Difference between embed_query and embed_passages?**

`embed_query` embeds a single short search query — typically called once per user request.

`embed_passages` embeds a list of document passages — called during indexing (potentially thousands of texts). It accepts `batch_size` to control how many texts are sent per API call (controlling memory and throughput).

The distinction also matters for asymmetric models: query and passage embeddings may use different internal representations. Our `OllamaEmbeddingsClient` implements both with the same underlying `_embed()` call (symmetric model), but maintains the interface for drop-in compatibility with `JinaEmbeddingsClient`.


In OllamaEmbeddingsClient:

async def embed_query(self, query: str) -> list:
    return self._embed([query])[0]       # single string → single vector

async def embed_passages(self, texts: list, batch_size: int = 100) -> list:
    # Batches in groups of 100, returns list of vectors
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        all_embeddings.extend(self._embed(texts[i:i+batch_size]))
    return all_embeddings
---

**Q8. Batching embedding calls efficiently.**

Instead of calling the API once per text (N round trips), batch texts into groups and send multiple texts per call.

```python
async def embed_passages(self, texts: list, batch_size: int = 100) -> list:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(self._embed(batch))
    return all_embeddings
```

Trade-offs: larger batches = fewer round trips = higher throughput, but higher memory usage per call and longer timeout risk. Ollama's `/api/embed` handles batches well up to ~100 texts. For production indexing of thousands of papers, batch_size=32–64 is safer to avoid OOM on the model server.



Batching reduces the number of HTTP round-trips to the embedding server. Instead of 77 calls (one per chunk), we send chunks in groups of 100 (one call for the whole corpus).

Trade-offs:

Larger batches: fewer round-trips, higher throughput, but higher memory usage per call and longer timeout risk
Smaller batches: more resilient (one failure loses less work), lower peak memory
Optimal batch size depends on model and server — 32–100 is typical for sentence embedding models
In HybridIndexingService, embed_passages(texts, batch_size=100) is called with all chunks from one paper at once. For 77 chunks across 3 papers, this is roughly 1–2 API calls total.

---

**Q9. Interface methods needed for OllamaEmbeddingsClient to be a drop-in.**

We inspected `JinaEmbeddingsClient` and matched every method signature:

```python
async def embed_query(self, query: str) -> List[float]
async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]
async def generate_embeddings(self, texts: List[str]) -> List[List[float]]  # alias
async def close(self) -> None
self.embedding_model: str  # attribute used in chunk metadata
self.EMBEDDING_DIMENSION: int = 768  # class-level constant
```

The hardest mismatch to catch was `batch_size` — `HybridIndexingService` calls `embed_passages(texts=..., batch_size=...)`, which caused `TypeError` until we matched the parameter name exactly.



JinaEmbeddingsClient's interface (discovered via inspect.signature + repeated AttributeError debugging):

async def embed_query(self, query: str) -> List[float]
async def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]
async def close(self)
Additional method needed for the agentic service:

async def generate_embeddings(self, texts: list) -> list  # alias for embed_passages
The async def requirement was critical — HybridIndexingService does await embeddings_client.embed_passages(...), so sync methods cause TypeError: object list can't be used in 'await' expression.

We also needed embedding_model as an attribute (for chunk metadata tagging) and EMBEDDING_DIMENSION = 768 as a class constant.

---

**Q10. Why must embed_query and embed_passages be async def?**

The `HybridIndexingService` and `ask.py` router call these with `await`:
```python
query_embedding = await embeddings_service.embed_query(request.query)
```

If the method is a plain `def` (synchronous), `await` on it returns a coroutine-like object, not a list — causing `TypeError: object list can't be used in 'await' expression`. In FastAPI's async request handlers, all I/O should be async anyway to avoid blocking the event loop.

Our `OllamaEmbeddingsClient` uses sync `requests.post()` internally but wraps it in `async def`, which works for our scale (one request at a time) but would block the event loop under high concurrency. True async would use `httpx.AsyncClient`.




FastAPI runs inside an async event loop (uvicorn + asyncio). When a route handler is async def, all I/O operations it awaits must themselves be awaitable coroutines.

HybridIndexingService calls await self.embeddings_client.embed_passages(...). If embed_passages is a regular def, await on it doesn't yield a coroutine — it returns the list directly, and await on a non-awaitable raises TypeError: object list can't be used in 'await' expression. That's exactly what we hit.

Making them async def (even though they internally use sync requests.post) wraps the function body in a coroutine. The sync HTTP call blocks the event loop briefly — acceptable for small-scale use, but for production you'd use httpx.AsyncClient for truly non-blocking I/O.

---

**Q11. Benchmarking embedding throughput.**

```python
import time, requests

texts = ["sample text"] * 100  # 100 texts
batch_sizes = [1, 8, 16, 32, 64, 100]

for batch_size in batch_sizes:
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        requests.post("http://10.2.2.183:11434/api/embed",
                     json={"model": "jina/jina-embeddings-v2-base-en", "input": batch})
    elapsed = time.time() - start
    print(f"batch_size={batch_size}: {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/s)")
```

Key metrics: texts/second, latency per batch, GPU utilization. On a T4 GPU, jina-v2 should do ~500–1000 texts/second at batch_size=32.

---

**Q12. What is qwen2.5:3b-instruct?**

`qwen2.5` is the model family from Alibaba's Qwen team. `3b` means approximately 3 billion parameters. `instruct` means the model has been fine-tuned for instruction following using RLHF/DPO — it responds to prompts like "Answer this question based on context" rather than just completing text.

Base (`qwen2.5:3b`): raw language model, good at completion tasks. Instruct-tuned variant: much better at question answering, following specific formats, RAG tasks. For our `/api/v1/ask` endpoint, instruct is essential — the base model might ignore the "Answer ONLY from the context" instruction.

---

**Q13. Strategies for larger models under latency constraints.**

- **Quantization**: use 4-bit (Q4_K_M) instead of 8-bit or FP16 — our qwen2.5:3b-instruct is Q4_K_M, reducing size ~4x with ~5% quality loss
- **Streaming**: start returning tokens immediately rather than waiting for full generation — perceived latency drops dramatically (first token in 0.3s vs 15s total)
- **Caching**: Redis exact-match cache means common queries never hit the LLM at all
- **Smaller context**: limit `top_k` chunks passed to LLM; fewer tokens = faster generation
- **Speculative decoding**: use a small draft model to propose tokens, large model to verify — not yet in our Ollama setup
- **GPU upgrade**: T4 → A100 gives ~5–8x throughput improvement



Streaming responses: start sending tokens immediately — first token in ~0.3s even if full response takes 15s, improving perceived latency
Quantization: use Q4 or Q5 instead of Q8 or FP16 — roughly 2x faster generation with acceptable quality loss
Speculative decoding: use a small draft model to propose tokens, large model to verify — can give 2–4x speedup
Caching: Redis exact-match cache means repeat queries are ~0ms regardless of model size
Reduce top_k: fewer chunks = shorter prompt = faster generation. We used top_k=1 for the cache test to see the speed difference
Batching requests: if multiple users ask simultaneously, batch their prompts (Ollama supports this via concurrent requests)
Smaller context: chunk the prompt itself if it's very long — send only the most relevant chunk

---

## 4. FastAPI & API Design

**Q1. Dependency injection in FastAPI.**

Dependency injection (DI) is a pattern where a function's dependencies are provided externally rather than created inside it. FastAPI's Depends() declares that a parameter should be resolved by calling another function first.

def get_opensearch_client(request: Request) -> OpenSearchClient:
    return request.app.state.opensearch_client

async def ask_question(
    request: AskRequest,
    opensearch_client: Annotated[OpenSearchClient, Depends(get_opensearch_client)],
):
    ...
When /ask is called, FastAPI sees the Depends(get_opensearch_client) annotation, calls get_opensearch_client(request), and injects the result as opensearch_client. This enables:

Testability (swap real client for mock)
Reuse (same dependency shared across many routes)
Lifecycle management (database sessions, connection pools)
In our system, all services (OpenSearchDep, EmbeddingsDep, OllamaDep, LangfuseDep, CacheDep) are provided via Depends() reading from request.app.state (populated during lifespan startup).

Benefits: testable (replace dependencies with mocks), reusable, handles lifecycle. FastAPI calls each dependency function once per request and caches the result within that request.

---

**Q2. lru_cache on get_settings() — why and risks.**

@lru_cache memoizes the function — the first call executes and the result is cached; subsequent calls return the cached result without re-executing.

@lru_cache
def get_settings() -> Settings:
    return Settings()
Benefits: Settings() reads all environment variables and validates them — doing this on every request would be wasteful. Cache it once.

Problems:

Env var changes after first call are ignored: if you os.environ["POSTGRES_DATABASE_URL"] = "..." after get_settings() was already called, the cached Settings object still has the old value — which is exactly the bug we hit repeatedly in Jupyter notebooks. Fix: set env vars before any src.config import, or call get_settings.cache_clear().
Test isolation: tests that need different settings must cache_clear() between tests
Module-level singleton: because Python caches at the module level, a cached settings object in a multi-worker uvicorn deployment is shared across workers (usually fine, but be aware)

---

**Q3. Pydantic BaseModel.**

Pydantic is a data validation library. `BaseModel` provides:
- **Validation**: field types are enforced, invalid values raise `ValidationError`
- **Serialization**: `.model_dump()` converts to dict, `.model_dump_json()` to JSON
- **Documentation**: FastAPI uses Pydantic models to auto-generate OpenAPI schemas

```python
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(3, ge=1, le=10)
    model: str = Field("qwen2.5:3b-instruct")
    categories: Optional[List[str]] = None
```

When FastAPI receives a POST body, it passes the JSON to AskRequest(**json_data). Pydantic:

Validates types (e.g. top_k must be an int)
Validates constraints (ge=1, le=10 means 1 ≤ top_k ≤ 10)
Converts types (string "3" → int 3 automatically)
Raises 422 Unprocessable Entity with a clear error message if validation fails
Generates OpenAPI schema automatically from the model

FastAPI automatically validates incoming JSON against this schema, returning 422 for invalid requests.

---

**Q4. Annotated in FastAPI.**

`Annotated[T, metadata]` attaches metadata to a type without changing the type itself (Python 3.9+). FastAPI interprets `Depends()`, `Query()`, `Body()` etc. placed in the metadata position:

```python
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
```

This is cleaner than the older `dep: OpenSearchClient = Depends(get_opensearch_client)` syntax — it keeps the type annotation pure and separates the FastAPI-specific wiring. It also works better with type checkers.

---

**Q5. lifespan context manager in FastAPI.**

lifespan is an async context manager that handles startup and teardown logic for the entire application — runs once when the server starts, not on every request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: runs before the first request
    app.state.database = make_database()
    app.state.opensearch_client = make_opensearch_client()
    app.state.embeddings_service = OllamaEmbeddingsClient(base_url=settings.ollama_host)
    yield  # <-- server is running and serving requests here
    # TEARDOWN: runs after the last request (graceful shutdown)
    app.state.database.teardown()
    langfuse_tracer.shutdown()
```

Startup: create DB connections, initialize clients, validate service health. Teardown: flush LangFuse traces, close DB connections, release resources. Replaces the deprecated `@app.on_event("startup")` pattern.
app = FastAPI(lifespan=lifespan)
What goes in startup: database connections, service clients, connection pool warmup, index validation. What goes in teardown: closing connections, flushing buffers (LangFuse flush), releasing resources.


---

**Q6. StreamingResponse and SSE.**

StreamingResponse returns chunks of data progressively as they're generated, rather than waiting for the full response.

For Server-Sent Events (SSE), the format is:

data: {"chunk": "The diffusion "}\n\n
data: {"chunk": "model works"}\n\n
data: {"done": true, "answer": "The diffusion model works..."}\n\n
Each event is prefixed with data: and terminated with two newlines. The client (Gradio, or requests.post(..., stream=True)) reads line by line via response.iter_lines().

async def event_generator():
    async for chunk in ollama_client.generate_stream(...):
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    yield f"data: {json.dumps({'done': True})}\n\n"

return StreamingResponse(event_generator(), media_type="text/plain")
The media_type="text/plain" is used instead of "text/event-stream" to avoid browser caching issues with SSE — the Gradio client parses it the same way.

Server-Sent Events (SSE) is a protocol where the server sends `data: <payload>\n\n` lines. The client (`requests.post(..., stream=True)`) reads these line-by-line. This is how our `/api/v1/stream` endpoint delivers LLM tokens to the Gradio UI as they're generated, giving a "typing" effect rather than waiting 15 seconds for the full answer.

---

**Q7. app.state vs module-level globals.**

app.state is the recommended pattern for FastAPI applications. It stores service instances on the application object, making them:

Accessible via request.app.state.opensearch_client in route handlers
Created in lifespan startup (proper lifecycle)
Testable (can create a test app with mock state)
Scoped to the app instance (not a Python module global, so no cross-test pollution)
Module-level globals are simpler but problematic:

Created at import time (not when the app starts) — can't use config values not yet loaded
Persist across tests unless explicitly reset
Not visible in OpenAPI docs
Race conditions in multi-threaded startup
Q8. 'LangfuseTracer' object has no attribute 'trace_rag_request' — how to debug?

1. Read the full traceback — note the file and line (`src/services/langfuse/tracer.py:21`)
2. Use `inspect.signature()` or `dir(obj)` to see what methods actually exist on the object
3. Search the codebase for where the method should be defined: `grep -rn "def trace_rag_request" src/`
4. Hypothesis: method doesn't exist on `LangfuseTracer` at all — `RAGTracer` was calling a method on the wrong object (calling `self.tracer.trace_rag_request` when `trace_rag_request` was defined on `RAGTracer` itself)
5. Fix: replace with the correct v3 SDK method (`self.tracer.client.start_as_current_span(...)`)

Key lesson: `AttributeError` on a service object almost always means a version mismatch, wrong class, or a method that was renamed/moved between SDK versions.



This is an AttributeError — you're calling a method that doesn't exist on the object.

Systematic debugging approach:

Read the traceback — file path and line number tell you exactly where the call is: src/services/langfuse/tracer.py line 21
Inspect the object's real API: print(dir(langfuse_tracer)) or import inspect; inspect.getmembers(langfuse_tracer, predicate=inspect.ismethod)
Check the source: type src\services\langfuse\client.py — does trace_rag_request exist there?
Check git history — was it renamed or removed in a recent commit?
Fix: either call the existing method (in our case create_trace → then update to v3's start_as_current_span) or add the missing method
Root cause in our case: RAGTracer.trace_request called self.tracer.trace_rag_request(...) — a method that never existed on LangfuseTracer. It was calling a nonexistent method on the wrong object.

---

**Q9. How openapi.json helps debug missing endpoints.**

FastAPI automatically generates OpenAPI 3.0 documentation at /openapi.json. It lists every registered route with its method, path, request schema, and response schema.

When we saw 404 on /api/v1/ask, we ran:

curl -s http://localhost:8000/openapi.json | python -c "import json,sys; print('\n'.join(json.load(sys.stdin)['paths'].keys()))"
And got only:

/api/v1/ping
/api/v1/health
/api/v1/papers/
/api/v1/papers/{arxiv_id}
/api/v1/hybrid-search/
This confirmed /ask and /stream weren't registered at all — main.py wasn't calling app.include_router(ask_router). The schema is ground truth for what the running app exposes, regardless of what's in code files.
This lists every registered route. We used this to confirm `/api/v1/ask` and `/api/v1/stream` were genuinely missing (not registered in `main.py`) rather than returning 404 for other reasons like path mismatch. Once we confirmed the routes were absent, we knew the fix was `app.include_router(ask_router, prefix="/api/v1")`, not a URL typo.

---

**Q10. 404 on an endpoint that exists in code.**

Likely causes, in order of probability:
1. Router not registered in `main.py` (`app.include_router(...)` call missing)
2. Router registered with wrong prefix (e.g. `/api/v1/ask/stream` instead of `/api/v1/stream`)
3. Container running a stale image (code changes not rebuilt/redeployed)
4. Request hitting wrong host/port (e.g. nginx routing, port mismatch)
5. HTTP method mismatch (GET vs POST)

Diagnostic: check `openapi.json` to see if the route is registered at all, then check the exact path and method.

---

**Q11. make_agentic_rag_service() got unexpected keyword argument 'model'.**

Cause: get_agentic_rag_service in dependencies.py was calling:

make_agentic_rag_service(
    opensearch_client=opensearch,
    ollama_client=ollama,
    embeddings_client=embeddings,
    langfuse_tracer=langfuse,
    model=settings.ollama_model,  # ← this doesn't exist in the factory signature
)
But make_agentic_rag_service only accepts opensearch_client, ollama_client, embeddings_client, langfuse_tracer, top_k, and use_hybrid. No model parameter.

Fix: remove model=settings.ollama_model from the dependency call. The model should be passed per-request (from AskRequest.model) inside the LangGraph nodes at generation time, not baked in at service construction.

def get_agentic_rag_service(opensearch, ollama, embeddings, langfuse):
    return make_agentic_rag_service(
        opensearch_client=opensearch,
        ollama_client=ollama,
        embeddings_client=embeddings,
        langfuse_tracer=langfuse,
    )

---

**Q12. docker compose stop vs down vs down -v.**

docker compose stop: stops running containers (SIGTERM → SIGKILL) but keeps containers and volumes intact. docker compose start can restart them.

docker compose down: stops containers AND removes containers and networks. Volumes are preserved. Data survives.

docker compose down -v: stops and removes containers, networks, AND named volumes. All persisted data is deleted. This is what caused the LangFuse schema issue — the Postgres volume was wiped.

Use down when: recreating containers with new config. Use down -v when: you need a completely clean state (development reset). Use stop when: temporarily pausing services, plan to restart later.

---

**Q13. Why editing .env doesn't affect a running container.**

Environment variables are injected into a container at creation time (when docker compose up runs). They're baked into the container's runtime environment as a snapshot of .env at that moment.

Editing .env on disk changes the file but not the running container's environment. To pick up changes:

docker compose up -d --force-recreate api — recreates the container from scratch with new env vars
OR docker compose stop api && docker compose up -d api — same effect
Note: docker compose restart api is NOT sufficient — it just restarts the process inside the existing container, which still has the old env vars
Also: if compose.yml has an explicit environment: block setting the same key as .env, the compose block wins regardless of .env changes.

---

## 5. LangFuse Observability

**Q1. LLM observability — why it matters.**

LLM observability tracks what happens inside AI pipelines: what prompt was sent, what was retrieved, what the model generated, how long each step took, and what the final output was. Without it, debugging a bad RAG answer is like debugging a black box.

In production RAG: you need to see if retrieval was relevant (did we find the right chunks?), if the prompt was well-formed, if the LLM hallucinated, and which queries are slow or failing. LangFuse provides trace trees showing every step, latency breakdown, costs, and user feedback correlation.

---

Why it matters for RAG specifically:

Debugging: when a user gets a bad answer, you need to see which chunks were retrieved, what prompt was sent, what the LLM returned
Latency profiling: is slowness from embedding? search? generation? LangFuse shows each span's duration
Quality trending: track answer quality scores over time to catch degradation
Cost tracking: generation observation records token usage — you can calculate cost per query
Audit trail: in regulated domains, you need to prove what context the AI used to generate an answer
LangFuse provides all of this via traces (one per request) containing nested spans (one per pipeline step) and generation observations (one per LLM call).

**Q2. Trace vs span vs generation in LangFuse.**

- **Trace**: the top-level container for a complete request (one user question = one trace). Has input (the query) and output (the final answer).
- **Span**: a timed step within a trace — "query embedding", "search retrieval", "prompt construction". Has input/output, duration, and nests under the trace or another span.
- **Generation**: a special span specifically for LLM calls. Tracks model name, token counts, latency, and cost estimates in addition to input/output.

In our system: one trace (`rag_request`) contains spans (`query_embedding`, `search_retrieval`, `prompt_construction`) and a generation observation (`llm_generation` with `model=qwen2.5:3b-instruct`).

Hierarchy in our system:

TRACE: rag_request
  SPAN: query_embedding
  SPAN: search_retrieval
  SPAN: prompt_construction
  GENERATION: llm_generation (model: qwen2.5:3b-instruct)
Q3. LangFuse v2 vs v3 Python SDK — how does the tracing API differ?

**Q3. LangFuse v2 vs v3 Python SDK.**

v2 (old, what the codebase was written for):

trace = client.trace(name="rag_request", input={"query": q})
span = trace.span(name="retrieval", input={...})
span.end()
trace.update(output={"answer": "..."})
Explicit objects passed around. trace(), span(), generation() methods on the client.

v3 (current, OpenTelemetry-based):

with client.start_as_current_span(name="rag_request", input={"query": q}) as span:
    with client.start_as_current_span(name="retrieval", input={...}) as inner_span:
        ...
    client.update_current_trace(output={"answer": "..."})
Context-manager based. Spans nest via Python context (no explicit parent passing). The first span opened becomes the trace root automatically. Methods: start_as_current_span, start_as_current_generation, update_current_span, update_current_trace.


---

**Q4. OpenTelemetry-based and context propagation.**

OpenTelemetry (OTel) is an industry standard for observability. LangFuse v3 uses Python's `contextvars.ContextVar` to track the "current active span". When you open a `with client.start_as_current_span("foo"):` block, that span is stored in the context variable. Any span opened inside that block automatically finds the parent via the context variable, creating a hierarchy without explicitly passing parent references.

When you call client.start_as_current_span(...), it:

Creates a span
Sets it as the "current span" in a Python ContextVar
Any nested start_as_current_span calls automatically see the parent span via the ContextVar
When the with block exits, the span ends and the parent span is restored in context
This is why v3's RAGTracer doesn't pass trace=trace to child methods — the context propagation handles parent-child relationships automatically.

---

**Q5. flush_at and flush_interval.**

LangFuse batches trace events before sending them to the server to reduce network calls.

`flush_at=15`: send a batch when 15 events have accumulated.
`flush_interval=1.0`: send any pending events every 1 second regardless of count.

`flush()` forces immediate send of all pending events — critical before process shutdown to avoid losing the last few traces. In our `RAGTracer.trace_request`, we call `self.tracer.flush()` in the `finally` block after each request.

---

**Q6. Prisma migration and the v2→v3 schema conflict.**

Prisma is an ORM used by LangFuse to manage its PostgreSQL schema. Migrations are versioned SQL scripts that evolve the schema — adding tables, columns, indexes.

What happened: we had a LangFuse v2 container running previously (from langfuse/langfuse:2), which created a v2 schema in the langfuse_v3_postgres_data volume. When we switched to v3 (langfuse/langfuse:3 + langfuse-worker:3), v3 tried to apply its migrations on top of a partially-migrated v2 schema.

The error: Invalid prisma.evalTemplate.upsert() — The column 'type' does not exist — v3 expected a type column in evalTemplate table that v2's schema never added.

Fix: wipe the LangFuse Postgres volume (docker volume rm arxiv-paper-curator_langfuse_v3_postgres_data) and let v3 run all migrations fresh on a clean database.

---

**Q7. start_as_current_span vs start_span.**

`start_as_current_span(name)`: a context manager that sets the span as the "current active span" in the context variable. Child spans opened inside the `with` block automatically become children of this span. This is the v3 idiomatic way.

`start_span(name)`: creates a span but doesn't set it as current in context. You get back a span object you can manually attach children to. Less common in v3; used when you need more manual control.

For our RAGTracer, we use `start_as_current_span` via `LangfuseTracer.start_span()` so that nesting (query_embedding inside rag_request, etc.) happens automatically.

---

**Q8. output: null in LangFuse — cause and debugging.**

output: null means the trace was created (input captured) but the generation step never set the output on the trace. Causes:

Exception before end_request: if an error occurs mid-pipeline, end_request(trace, response, ...) is never called → output stays null. Check the API error logs.
Wrong Ollama host → 404 → exception swallowed: our case — OLLAMA_HOST=http://ollama:11434 returned 404, the exception was caught and re-raised as HTTP 500, but end_request never ran.
Tracing method doesn't exist: 'Langfuse' object has no attribute 'trace' — the span creation failed silently (our try/except in trace_request), so trace=None, and end_request(None, ...) is a no-op.
Debugging: check docker compose logs api after making a request. If you see any WARNING - Tracing error or ERROR, that's where the trace broke.
Root cause: the `/api/v1/ask` endpoint was returning a 500 error due to the `RAGTracer.trace_request` bug (`'Langfuse' object has no attribute 'trace_rag_request'`). The trace was created (LangFuse SDK creates it before the error), but the `end_request()` call (which sets the output) was never reached because an exception was thrown first.

Debug steps: check `docker compose logs api` for the actual exception traceback, not just the HTTP 500 status. The logs showed the `AttributeError` immediately.

---

**Q9. localhost:3000 inside Docker container.**

Inside a Docker container, localhost refers to the container itself — not the host machine, not other containers. Port 3000 isn't listening inside the rag-api container, so all HTTP requests to http://localhost:3000 fail with connection refused, and LangFuse silently drops events.

The correct value is the Docker service name: http://langfuse-web:3000. Docker's internal DNS resolves langfuse-web to the container's IP on the rag-network bridge network.

From your host machine browser: http://localhost:3001 (external port). From other containers on the same network: http://langfuse-web:3000 (internal service name + container port).

---

**Q10. Single underscore vs double underscore in pydantic-settings.**

pydantic-settings uses `env_nested_delimiter` to map nested settings classes:

```python
class LangfuseSettings(BaseSettings):
    host: str = "http://localhost:3000"

class Settings(BaseSettings):
    langfuse: LangfuseSettings  # nested

    model_config = SettingsConfigDict(env_nested_delimiter="__")
```

With `env_nested_delimiter="__"`:
- `LANGFUSE__HOST=http://langfuse-web:3000` → `settings.langfuse.host = "http://langfuse-web:3000"` ✓ (read)
- `LANGFUSE_HOST=http://langfuse-web:3000` → not matched by pydantic nested settings, ignored ✗

This caused a persistent bug: `compose.yml` set `LANGFUSE_HOST` (single underscore, ignored by pydantic) while `.env` set `LANGFUSE__HOST=http://localhost:3000` (wrong value, but correctly read). Fix: align both to `LANGFUSE__HOST` with the correct value.

---

**Q11. Verifying LangFuse actually receives traces.**

1. Check startup logs: `docker compose logs api | findstr langfuse` → look for `"Langfuse v3 tracing initialized"` (good) vs `"Langfuse tracing disabled"` (credentials missing/wrong)
2. Check for connection errors in request logs: `"Tracing error in trace_request: ..."` means the SDK call failed
3. Test health endpoint from inside the container: `docker exec -it rag-api python -c "import httpx; print(httpx.get('http://langfuse-web:3000/api/public/health').status_code)"`
4. Make a test request, then check LangFuse UI → Traces. If no trace appears, check `LANGFUSE__HOST`, `LANGFUSE__PUBLIC_KEY`, and `LANGFUSE__SECRET_KEY` match the project's current API keys.
5. Enable `LANGFUSE__DEBUG=true` for verbose SDK logging showing every HTTP call to LangFuse.

---

Check container logs for initialization:
docker compose logs api | grep -i langfuse
# Look for: "Langfuse v3 tracing initialized" vs "Langfuse tracing disabled or missing credentials"
Check for connection errors:
docker compose logs api | grep -i "tracing error\|langfuse error"
Test connectivity from inside the container:
docker exec -it rag-api python -c "import httpx; print(httpx.get('http://langfuse-web:3000/api/public/health', timeout=5).status_code)"
# Should return 200
Make a test request and check immediately:
curl -X POST http://localhost:8000/api/v1/ask -H "Content-Type: application/json" \
  -d '{"query": "test", "model": "qwen2.5:3b-instruct"}'
# Then refresh http://localhost:3001 — new trace should appear within 1-2 seconds (flush_interval=1.0)
Use LangFuse's own auth check:
from langfuse import Langfuse
lf = Langfuse(public_key="pk-...", secret_key="sk-...", host="http://langfuse-web:3000")
lf.auth_check()  # Returns True if credentials are valid and host is reachable

**Q12. LangFuse CallbackHandler vs manual spans.**

`CallbackHandler` is a LangChain/LangGraph integration that automatically hooks into LLM calls, chain executions, and tool calls via LangChain's callback system. You attach it when invoking a graph: `graph.invoke(input, config={"callbacks": [handler]})`. Every LLM call inside the graph is automatically traced.

handler = langfuse_tracer.get_callback_handler(
    trace_name="agentic_rag",
    user_id="api_user",
    session_id="session_123"
)

# LangGraph automatically calls handler callbacks at each node
result = graph.invoke(input, config={"callbacks": [handler]})
Vs manual span creation (RAGTracer): you explicitly create and end spans at each step. More control, more code.

CallbackHandler is better for LangGraph because the graph structure (nodes, edges, retries) maps naturally to LangFuse's trace tree — every node invocation becomes a nested span automatically.


In our system: `trace_langgraph_agent` returns a `CallbackHandler` for LangGraph, while the standard `/ask` route uses `RAGTracer` with manual spans. The agentic route uses the callback approach; the standard route uses manual tracing.

---

**Q13. score_current_trace / create_score API.**

`create_score` attaches a numeric evaluation score to a trace — used for logging model quality metrics, user feedback, or automated evals:

```python
langfuse_client.create_score(
    trace_id="some-trace-id",
    name="relevance",
    value=0.85,
    comment="Retrieved chunks were highly relevant"
)
```

Use cases: logging guardrail scores (our system records `"Validated query scope (score: 50/100)"`), storing user thumbs-up/thumbs-down feedback, automated evaluation of answer quality, A/B testing embedding models by comparing average scores.

---

## 6. LangGraph & Agentic RAG

**Q1. LangGraph vs LangChain chains.**

LangChain: a linear chain of components (prompt → LLM → parser). Good for simple, fixed pipelines. No branching or loops.

LangGraph: a state machine built on a directed graph. Nodes are functions (Python callables), edges define transitions, conditional edges allow branching based on the current state. Supports cycles (loops), enabling retry logic, multi-step reasoning, and tool use.

Nodes are Python functions (processing steps)
Edges define the flow between nodes
Conditional edges allow branching based on node output (e.g., "if guardrail fails → go to rejection node; else → go to retrieval node")
Cycles are possible (e.g., retrieval → grading → rewrite → retrieval again)
State is shared across all nodes via a typed state dict
LangGraph is better for agentic RAG because real RAG pipelines aren't linear — they need to decide whether to retrieve, whether retrieved docs are good enough, whether to rewrite and retry, and how to handle scope violations.

---

**Q2. Nodes, edges, and conditional routing.**

Node: an async Python function that receives the current graph state and returns a (partial) state update.

async def guardrail_node(state: RAGState) -> dict:
    score = await llm.evaluate_scope(state["query"])
    return {"scope_score": score, "is_in_scope": score >= 60}
Edge: a directed connection from one node to another. All edges from a node fire after it runs.

Conditional edge: a router function that decides which node to go to next based on state.

def route_after_guardrail(state: RAGState) -> str:
    return "retrieval" if state["is_in_scope"] else "rejection"

graph.add_conditional_edges("guardrail", route_after_guardrail, {
    "retrieval": "retrieve_documents",
    "rejection": "generate_rejection"
})
Q3. What is GraphConfig in a LangGraph service?

**Q3. GraphConfig.**

`GraphConfig` is a dataclass that configures the LangGraph at construction time:

```python
@dataclass
class GraphConfig:
    top_k: int = 3
    use_hybrid: bool = True
```

These are service-level defaults, not per-request. Per-request parameters (like the user's specific `top_k` or `model`) are passed through the agent state at invocation time. We hit a bug where `model` was being passed to `make_agentic_rag_service()` (service-level construction) instead of being threaded through the state per-request.

---

**Q4. Guardrail node and query scope validation.**

A guardrail node is the first node in the LangGraph pipeline that decides whether the incoming query is appropriate for the system to answer.

Query scope validation: the node uses the LLM to score how relevant the query is to the system's intended domain (CS/AI/ML research papers in our case).

Input query: "What is a dog?"
LLM prompt: "Rate this query's relevance to academic AI/ML research on a scale of 0-100. Query: 'What is a dog?'"
LLM response: 50
Decision: 50 < 60 threshold → REJECT
Result: retrieval_attempts: 0, answer is a polite "this is outside my domain" message. No expensive retrieval or generation wasted.

---

**Q5. Query rewriting and multiple retrieval attempts.**

After the first retrieval, a document grading node evaluates whether the retrieved chunks are relevant enough to answer the question. If they're not:

A query rewriting node is invoked — it uses the LLM to generate a better search query (e.g., more specific terms, different synonyms, more precise scope)
Retrieval runs again with the rewritten query
This can repeat up to max_attempts times (typically 2-3)
Why: the first query might use informal language that doesn't match technical paper vocabulary. "Tell me about code repair stuff" → rewritten to "diffusion model automatic code repair generation arXiv 2025" → better retrieval.

retrieval_attempts in the response tells you how many rounds were needed — 1 means first try worked, 2+ means rewriting was triggered.

**Q6. AgenticRAGService vs simple rag_ask() function.**

`rag_ask()` is a linear function: embed → retrieve → prompt → generate. Fixed behavior, no branching, no retry logic. Simple, predictable, fast.

Single function, no LangGraph
Linear: embed → search → prompt → generate
No guardrail, no query rewriting, no retry
Fast to develop, easy to debug
Good for: prototyping, notebooks, simple use cases
AgenticRAGService (production):

LangGraph graph with multiple nodes
Has guardrail (rejects out-of-scope), document grading, query rewriting, retry
Observability (LangFuse tracing at each node)
More complex but more robust
Good for: production API, untrusted user input, varied query types
Use rag_ask() for: quick validation that retrieval and generation work. Use AgenticRAGService for: production deployment where you need quality controls.

---

**Q7. retrieval_attempts: 0.**

The query was rejected by the guardrail node before any retrieval was attempted. Specifically, the scope score fell below the threshold (< 60 in our system).

This is the expected behaviour for out-of-scope queries like "What is a dog?" or "What's the weather?" — the system provides a helpful explanation of its scope limitations rather than hallucinating an answer or wasting compute on irrelevant retrieval.

---

**Q8. Passing per-request model to LangGraph constructed once.**

Since `AgenticRAGService` is created once at dependency injection time (not per-request), per-request parameters like `model` must be passed through the **agent state**, not the constructor:

# Service is constructed once (DI)
service = AgenticRAGService(opensearch_client, ollama_client, ...)

# Per-request invocation passes state
result = await service.run({
    "query": request.query,
    "model": request.model,      # per-request parameter in STATE
    "top_k": request.top_k,
    "use_hybrid": request.use_hybrid,
})
Each node reads state["model"] when it needs to make an LLM call. This is why removing model=settings.ollama_model from make_agentic_rag_service() was correct — the model belongs in the state, not the constructor.

---

**Q9. reasoning_steps field.**

reasoning_steps is a list of strings describing what the agent did at each stage of processing — a human-readable trace of the decision path.

["Validated query scope (score: 50/100)", "Generated answer from context"]
# or
["Validated query scope (score: 90/100)", "Retrieved 3 documents", "Generated answer from context"]
# or  
["Validated query scope (score: 85/100)", "Retrieved 3 documents", "Rewrote query (attempt 2)", "Retrieved 3 documents", "Generated answer from context"]
Each LangGraph node appends to this list as it executes:

state["reasoning_steps"].append(f"Validated query scope (score: {score}/100)")
This provides transparency about what the agent did — useful for debugging and for end users to understand why they got a particular answer or rejection.

---

**Q10. Adding a document grading/relevance filtering node.**

```python
def grade_documents_node(state: AgentState) -> AgentState:
    """Filter retrieved chunks to only truly relevant ones."""
    relevant_chunks = []
    for chunk in state.retrieved_chunks:
        prompt = f"Is this chunk relevant to '{state.query}'? Yes or No:\n{chunk['chunk_text']}"
        response = ollama_client.generate(model=state.model, prompt=prompt)
        if "yes" in response.lower():
            relevant_chunks.append(chunk)
    
    return {
        **state,
        "retrieved_chunks": relevant_chunks,
        "reasoning_steps": state.reasoning_steps + [f"Graded chunks: {len(relevant_chunks)}/{len(state.retrieved_chunks)} relevant"]
    }
```

Add this node between retrieval and generation, with a conditional edge: if `len(relevant_chunks) == 0` → trigger query rewriting; else → proceed to generation.

---


async def grade_documents_node(state: RAGState) -> dict:
    chunks = state["retrieved_chunks"]
    query = state["query"]
    
    grading_prompt = f"""Rate each document's relevance to: '{query}'
    Documents: {[c['chunk_text'][:200] for c in chunks]}
    Return JSON: {{"scores": [0-10, ...]}}"""
    
    response = await ollama_client.generate(model=state["model"], prompt=grading_prompt)
    scores = json.loads(response)["scores"]
    
    # Keep only chunks scoring >= 6
    relevant_chunks = [c for c, s in zip(chunks, scores) if s >= 6]
    
    return {
        "retrieved_chunks": relevant_chunks,
        "documents_graded": True,
        "should_rewrite": len(relevant_chunks) == 0
    }

# Add conditional edge after grading
graph.add_conditional_edges("grade_documents", 
    lambda s: "rewrite" if s["should_rewrite"] else "generate",
    {"rewrite": "rewrite_query", "generate": "generate_answer"}
)
**Q11. Latency trade-offs of agentic vs non-agentic.**

Non-agentic (`rag_ask`): 1 embedding call + 1 search + 1 generation = ~15s total. Fixed cost per request.

Agentic: guardrail (1 LLM call ~2s) + embedding + search + optional query rewrite (1 more LLM call ~2s) + generation (~12s) = 15–20s for in-scope queries, but only ~2s for out-of-scope rejections.
Embedding: ~100ms
Search: ~50ms
Generation: ~14s (dominant)
Agentic (LangGraph with guardrail, grading, optional rewriting):

Best case (in-scope, first retrieval good): ~17s (+2s for guardrail LLM call)
Rejection case (out-of-scope): ~3s (guardrail LLM call only, no generation)
Worst case (2 retrieval attempts with rewriting): ~30s (guardrail + 2× retrieval + rewrite + generate)
Trade-off: agentic adds 2–15s of latency but significantly improves quality, reduces hallucination on out-of-scope queries, and improves recall via query rewriting. For production, the quality gain usually justifies the latency cost.

---

## 7. PostgreSQL & SQLAlchemy
**Q1. What is SQLAlchemy and what is the difference between Core and ORM?**

SQLAlchemy is Python's most popular database toolkit. It has two layers:

Core: SQL Expression Language — construct SQL queries programmatically using Python objects, close to raw SQL. Better for complex queries and performance-critical code.

ORM (Object-Relational Mapper): maps Python classes to database tables. You work with Python objects; SQLAlchemy generates the SQL. Our Paper model is an ORM model.

# ORM (what we use)
papers = session.query(Paper).filter(Paper.raw_text.isnot(None)).all()

# Core equivalent
result = conn.execute(select(papers_table).where(papers_table.c.raw_text.isnot(None)))
We use ORM because it's cleaner for CRUD operations on the papers table and integrates better with our PaperRepository pattern.

**Q2. get_session() as a context manager.**

```python
with database.get_session() as session:
    papers = session.query(Paper).filter(Paper.raw_text.isnot(None)).all()
```

The context manager ensures the session is:
1. Created (connection checked out from pool)
2. Committed on success (changes persisted)
3. Rolled back on exception (no partial writes)
4. Closed on exit (connection returned to pool)

Without this, long-lived sessions leak connections, exhausting the pool. Manual `session.close()` calls are error-prone (missed on exceptions). Context managers make it automatic.


with database.get_session() as session:
    paper_repo = PaperRepository(session)
    stored_paper = paper_repo.upsert(paper_create)
    # session.commit() called on __exit__ if no exception
    # session.rollback() called on __exit__ if exception
    # session.close() called on __exit__ regardless
Why lifecycle management matters:

Connections are finite: unclosed sessions hold database connections indefinitely, exhausting the connection pool
Transactions: uncommitted changes aren't visible to other sessions (isolation). commit() must be called explicitly or automatically on context exit
Rollback on error: if an exception occurs mid-session, rollback prevents partial writes
Memory: ORM sessions cache loaded objects in an identity map — long-lived sessions accumulate memory

---

**Q3. What is an upsert?**

An upsert (update + insert) inserts a new record if it doesn't exist, or updates it if it does — based on a unique key (in our case, arxiv_id).

# Conceptual upsert logic in paper_repo.upsert()
existing = session.query(Paper).filter_by(arxiv_id=paper_create.arxiv_id).first()
if existing:
    # Update fields
    existing.title = paper_create.title
    existing.raw_text = paper_create.raw_text
    session.commit()
    return existing
else:
    new_paper = Paper(**paper_create.dict())
    session.add(new_paper)
    session.commit()
    return new_paper
In PostgreSQL, this can be done efficiently with INSERT ... ON CONFLICT (arxiv_id) DO UPDATE SET .... Upserts are important for our pipeline because the same arXiv paper might be fetched multiple times (re-runs, different date ranges) and we want idempotent storage.

---

**Q4. lru_cache on make_database() — risks and benefits.**

@lru_cache on make_database() creates a single PostgreSQLDatabase instance shared across the entire application.

Benefits:

Connection pool is initialized once (expensive operation)
All requests share the same pool — efficient use of connections
Settings are read once — consistent throughout the app lifetime
Risks:

If database credentials change (e.g., password rotation), the cached database object uses stale credentials until restart
Tests that need fresh database state can't easily get a new instance without cache_clear()
In multi-process deployments (multiple uvicorn workers), each process has its own cache — connection pools multiply


**Q5. What is connection pooling and why does it matter?**

**Q5. Connection pooling.**

Creating a new database connection for every request is expensive (~5–50ms). Connection pooling maintains a set of pre-established connections and reuses them.

# In config.py
postgres_pool_size: int = 20      # max connections kept open
postgres_max_overflow: int = 0    # additional connections allowed above pool_size
pool_size=20: up to 20 simultaneous requests can be executing database queries. The 21st request waits for a connection to be returned to the pool.

max_overflow=0: strict — no temporary burst connections allowed. Set to 5–10 for applications with bursty traffic.

With FastAPI (async), a single worker can handle many concurrent requests, but database queries (synchronous, blocking) need dedicated connections from the pool. Size the pool based on your expected concurrent database operations, not total request concurrency.

Q6. filter() vs filter_by() in SQLAlchemy ORM?

filter_by(): keyword arguments with column names — simpler syntax for exact equality checks.

session.query(Paper).filter_by(arxiv_id="2508.11110v1").first()
filter(): accepts SQLAlchemy column expressions — more powerful, supports all comparison operators.

session.query(Paper).filter(Paper.raw_text.isnot(None)).all()
session.query(Paper).filter(Paper.published_date >= cutoff_date).all()
session.query(Paper).filter(Paper.arxiv_id.in_(["2508.11110v1", "2508.11112v1"])).all()
Use filter_by for simple equality on a known field. Use filter for everything else — inequalities, NULL checks, IN clauses, OR conditions, etc.

`filter()` is more powerful — it supports `!=`, `in_()`, `like()`, `isnot()`, compound conditions with `and_()`, `or_()`. `filter_by()` only supports equality. Use `filter()` for complex queries, `filter_by()` for simple lookups.

---

**Q7. Paper.raw_text.isnot(None).**

It generates a SQL WHERE raw_text IS NOT NULL clause. SQLAlchemy translates the Python .isnot(None) method on a column descriptor to the appropriate SQL NULL check.

# SQLAlchemy
session.query(Paper).filter(Paper.raw_text.isnot(None)).all()
# Generated SQL
SELECT * FROM papers WHERE raw_text IS NOT NULL
You can't use Python's != None for this because SQLAlchemy would generate WHERE raw_text != NULL which evaluates to NULL (always false) in SQL — you must use IS NOT NULL. SQLAlchemy provides .isnot(None) to enforce the correct SQL idiom.

---

**Q8. password authentication failed for user "rag_user" — causes and fixes?**

We hit this three times, each with a different cause:

1. **Wrong port**: `make_database()` defaulted to `localhost:5432` but the container was mapped to `localhost:5433`. Fix: set `POSTGRES_DATABASE_URL=postgresql://rag_user:rag_password@127.0.0.1:5433/rag_db` before any `get_settings()` call.

2. **lru_cache stale**: `get_settings()` was called before the env var was set, caching the wrong URL. Fix: set env var first, then `get_settings.cache_clear()`.

3. **IPv6 preference**: `localhost` resolved to `::1` (IPv6) but the Postgres container was only listening on IPv4. Fix: use `127.0.0.1` explicitly.

---

**Q9. Why must the env var be set before get_settings() is imported in a notebook var must be set before get_settings() import.**

Python's @lru_cache decorator caches the return value on the first call. get_settings() is called the moment any src.* module that imports it is first imported — often during from src.db.factory import make_database.

Timeline of the bug:

# Wrong order:
from src.db.factory import make_database    # ← triggers get_settings() internally
os.environ["POSTGRES_DATABASE_URL"] = "..."  # ← too late, cache already set
db = make_database()                         # ← uses cached, wrong settings
# Correct order:
import os
os.environ["POSTGRES_DATABASE_URL"] = "postgresql://rag_user:rag_password@127.0.0.1:5433/rag_db"
# ↑ Must happen BEFORE any src.* import
from src.db.factory import make_database    # ← get_settings() sees correct env var
db = make_database()                         # ← correct settings
---

**Q10. Docker port mapping — 5433 vs 5432.**


Docker port mapping publishes a container's internal port to a port on the host machine.

ports:
  - "5433:5432"  # host:container
5432: the port PostgreSQL listens on inside the container (standard Postgres port)
5433: the port exposed on your Windows/Mac machine
From your host (notebooks, psql CLI): connect to 127.0.0.1:5433 From other containers on the same Docker network: connect to postgres:5432 (service name + container port)

That's why the API container's POSTGRES_DATABASE_URL uses port 5432 (@postgres:5432/rag_db) while your notebook uses port 5433 (@127.0.0.1:5433/rag_db) — they're connecting from different network contexts.


---

## 8. Docker & Docker Compose

**Q1. Docker image vs container.**

Image: a read-only, layered snapshot of a filesystem — like a class definition. Built from a Dockerfile. Immutable. Can be pushed to a registry (Docker Hub, ACR).

Container: a running instance of an image — like an object instantiated from a class. Has its own writable layer, network interface, and process namespace. Multiple containers can run from the same image simultaneously.

docker build creates an image. docker run / docker compose up creates and starts containers from images.

**Q2. Docker volumes — named vs bind mount.**

Named volume (`postgres_data:/var/lib/postgresql/data`): managed by Docker, stored in Docker's storage area, survives container deletion. Best for databases and persistent data.

Bind mount (`./src:/opt/airflow/src`): maps a host directory into the container. Changes on the host immediately reflected in the container. Best for development (live code changes without rebuilding).

In our `compose.yml`: databases use named volumes (persist data), Airflow DAGs use bind mounts (edit DAGs without rebuilding the container).
We use named volumes for postgres_data, opensearch_data, ollama_data (persistence without caring about host path) and bind mounts for ./airflow/dags:/opt/airflow/dags (DAG files edited on host, visible in Airflow immediately).

---

**Q3. docker compose down -v vs docker compose down.**

docker compose down: stops and removes containers and networks. Named volumes are preserved. Data survives.

docker compose down -v: also removes all named volumes defined in compose.yml. All persisted data (PostgreSQL rows, OpenSearch indices, Redis cache, Ollama models) is permanently deleted.

Use down -v only for: complete fresh start (Week 2 clean slate), fixing corrupted volume data (LangFuse schema conflict), or before sharing a demo environment.

---

**Q4. Why docker compose stop doesn't free volumes.**

stop sends SIGTERM to the container process and waits for it to exit gracefully. The container still exists in a stopped state — its filesystem layer and volume attachments are intact. You can docker compose start to resume it.

Volumes are attached to containers, not to running processes. A stopped container still holds the volume mount reference, which is why docker volume rm fails with "volume is in use" even on a stopped container. You need docker compose rm (or docker rm) to actually delete the container object and release the volume reference.

---

**Q5. healthcheck and depends_on.**


A healthcheck is a command Docker runs periodically inside a container to determine if the service is ready — not just running, but actually functional.

postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_db"]
    interval: 5s
    timeout: 5s
    retries: 10
    start_period: 30s
depends_on with conditions:

api:
  depends_on:
    postgres:
      condition: service_healthy   # Wait until postgres healthcheck passes
    opensearch:
      condition: service_healthy
Without condition: service_healthy, depends_on only waits for the container to start — not for PostgreSQL to finish initializing. This caused the rag-api to crash on startup before Postgres was ready in earlier runs.


---

**Q6. container_name vs compose service name.**

Service name (compose key, e.g., api): used with docker compose commands: docker compose up -d api, docker compose logs api, docker compose stop api.

container_name (e.g., rag-api): the Docker container name — used with docker commands: docker exec -it rag-api bash, docker logs rag-api, docker stop rag-api.

Both identify the same container but in different contexts. The confusion caused our error: docker compose up -d rag-api failed because compose uses the service name (api), not the container name (rag-api).

---

**Q7. --force-recreate vs --build vs --no-cache.**

- `--force-recreate`: stops and removes existing containers, creates new ones even if config hasn't changed. Forces environment variable updates to take effect. Does NOT rebuild the image.
- `--build`: rebuilds images from Dockerfile before starting containers. Required after code changes.
- `--no-cache`: used with `build`, forces a full rebuild without using Docker's layer cache. Required when you need to guarantee fresh dependency installation (e.g. when `pyproject.toml` changes but Docker cache missed it).

For config-only changes (`.env`): `--force-recreate`. For code changes: `--build`. When build seems stale: `--build --no-cache`.

---

**Q8. Environment variable precedence — compose.yml environment: vs env_file:.**

compose.yml explicit environment: block wins over env_file:.

env_file:
  - .env              # LANGFUSE__HOST=http://localhost:3000 (from .env)
environment:
  - LANGFUSE__HOST=http://langfuse-web:3000   # ← this wins
This is the expected Docker behaviour and was useful for us: the compose.yml environment block provides container-specific overrides (service hostnames valid inside Docker network) on top of the .env file values (which use localhost addresses valid from the host machine).

The conflict that caused our issue: LANGFUSE__HOST wasn't in the compose.yml environment block (only LANGFUSE_HOST single-underscore was), so the .env value (http://localhost:3000) was read directly by pydantic-settings via the double-underscore key.

**Q9. Diagnosing an unhealthy but running container.**

Check the healthcheck command result:
docker inspect rag-api --format='{{json .State.Health}}' | python -m json.tool
# Shows last 5 healthcheck results and exit codes
Run the healthcheck command manually:
# For rag-api healthcheck: python -c "urllib.request.urlopen('http://localhost:8000/api/v1/health')"
docker exec -it rag-api python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
Check container logs for startup errors:
docker compose logs api --tail=50
Common causes: application crash-looping (ImportError, missing dependency), dependency not ready (DB not yet healthy when API tries to connect), healthcheck command itself is wrong.

---

**Q10. [Errno -3] Temporary failure in name resolution.**

This error means DNS resolution failed inside the container — it couldn't look up the hostname.

In our case: http://langfuse-web:3000 from inside rag-api container failed because langfuse-web wasn't running at the time (we had stopped it to remove the volume, but hadn't restarted it yet before testing).

DNS name resolution between containers works via Docker's internal DNS (127.0.0.11 inside each container), which maps service names to container IPs. This only works when:

Both containers are on the same Docker network (rag-network)
The target container is actually running
Fix: ensure langfuse-web is started (docker compose up -d langfuse-web) before making requests that need to reach it.

**Q11. Bridge network and why services need the same network.**

Docker creates an isolated virtual network for your compose project. Services on rag-network can reach each other by service name (Docker's internal DNS resolves them). Services NOT on the same network can't communicate at all — they're isolated.

networks:
  rag-network:
    driver: bridge
Bridge driver: creates a virtual Ethernet bridge on the host. Containers on the same bridge can communicate; they're isolated from containers on other bridges and from the host network (except through published ports).

All our services are on rag-network, which is why opensearch:9200, postgres:5432, langfuse-web:3000 work as hostnames within any container in the stack.

**Q12. localhost:3000 vs langfuse-web:3000 from inside a container.**

From inside rag-api container:
  localhost:3000          → rag-api container itself (nothing listening on 3000) ❌
  langfuse-web:3000       → langfuse-web container via rag-network DNS ✅
  
From your Windows host machine:
  localhost:3000          → nothing (published port is 3001, not 3000) ❌
  localhost:3001          → langfuse-web container via Docker port mapping ✅
The confusion arises because localhost means different things in different contexts:

Host machine: loopback to host's own network stack (127.0.0.1)
Container: loopback to the container's own network namespace
Service-to-service communication inside Docker always uses the compose service name, never localhost.

## 9. Airflow

**Q1. What is Airflow and what problem does it solve?**

Apache Airflow is a workflow orchestration platform — it schedules and monitors data pipelines defined as Python code. It solves the problem of reliably running complex multi-step jobs (like "fetch new arXiv papers every day, download PDFs, parse with Docling, store in PostgreSQL, index in OpenSearch") on a schedule with retry logic, error alerting, and execution history.

In our system, the `arxiv_paper_ingestion` DAG automates the weekly ingestion pipeline that we tested manually in the notebooks.

---

**Q2. DAG, task, and operator.**

**DAG** (Directed Acyclic Graph): the pipeline definition — a Python file describing which tasks run in what order, with what schedule.

**Task**: a single unit of work within a DAG — one step like "fetch papers from arXiv" or "parse PDF with Docling."

**Operator**: the class that executes a task. `PythonOperator` runs a Python function. `BashOperator` runs a shell command. `HttpSensor` polls an HTTP endpoint until it returns 200. We use `PythonOperator` to wrap our service functions.

PythonOperator: runs a Python function
BashOperator: runs a shell command
HttpOperator: makes an HTTP request
@task decorator: modern TaskFlow API that wraps Python functions as tasks
Our arxiv_paper_ingestion DAG uses Python tasks that call fetch_and_process_papers(), parse_pdfs(), and index_to_opensearch().

Q3. What does "paused" mean for a DAG?

A paused DAG is defined and valid but won't run on its schedule until unpaused. It's like a stopped cron job. You can still manually trigger it.

New DAGs in Airflow are paused by default (controlled by AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true). This is a safety feature to prevent accidental immediate execution when deploying new pipelines.

Our DAGs (arxiv_paper_ingestion, hello_world_week1) were paused because: (1) they were newly created, and (2) we hadn't finished testing the full pipeline before enabling automated scheduling.

---

**Q4. airflow dags list-import-errors.**

It lists any DAG files that Airflow couldn't import due to Python errors — syntax errors, missing imports, incorrect task configurations.

docker exec -it rag-airflow airflow dags list-import-errors
If this returns results, those DAGs won't appear in the UI and won't run. Common causes:

Missing Python package (we saw docling not installed in the Airflow container)
Syntax error in the DAG file
Import of a module that doesn't exist in the Airflow container's Python environment (even if it exists in the API container)
"No import errors" is the green light that your DAG files are syntactically valid.

**Q5. backfill vs manual trigger.**

Manual trigger (airflow dags trigger dag_id): runs the DAG once immediately with the current timestamp. For ad-hoc testing.

Backfill (airflow dags backfill -s 2025-01-01 -e 2025-01-31 dag_id): runs the DAG for all scheduled intervals between the start and end dates that haven't been run yet. For catching up on missed scheduled runs (e.g., the DAG was paused for a month and you want to process all the papers from that period).

For our arXiv ingestion: backfill would process papers from each day in the date range, rather than just "today's" papers.

---

**Q6. Docling not available in Airflow container.**

Each Docker service has its own Python environment. docling was installed in the rag-api container (via pyproject.toml and the API's Dockerfile) but the Airflow container has a separate image (arxiv-paper-curator-airflow built from airflow/Dockerfile).

Unless docling is explicitly added to the Airflow container's dependencies (in airflow/Dockerfile or airflow/requirements.txt), it won't be available there.

This is why list-import-errors showed a docling-related ImportError for the arxiv_paper_ingestion DAG — the DAG imports our PDF parsing code which imports docling, but docling isn't in the Airflow container.

Fix: add docling to the Airflow container's requirements, or restructure the DAG to call the FastAPI /ingest endpoint via HTTP instead of importing Python modules directly.

## 10. Python & Software Engineering

**Q1. @lru_cache risks on settings functions.**

@lru_cache on get_settings() caches the result indefinitely (no TTL). Risks:

Import-order sensitivity: env vars set after the first get_settings() call are ignored — the cache returns the original Settings object. We hit this in every Jupyter notebook session.
Test pollution: tests that modify environment variables between test cases need to call get_settings.cache_clear() explicitly, or they'll all see the same settings.
Process-level singleton: in multi-process deployments (Gunicorn with workers), each worker process has its own cache — settings are read once per worker. Usually fine, but be aware.
Memory leak risk: if the cached object holds expensive resources (connections, file handles), they're never released.
Mitigation: use lru_cache(maxsize=1) for clarity, document that cache_clear() is needed after env var changes, and use it only on truly immutable configurations.

---

**Q2. async def vs def — when is await required?**

def: synchronous function. Runs to completion without yielding control. Blocks the event loop if it does I/O.

async def: coroutine function. Can be paused with await to yield control back to the event loop, allowing other coroutines to run.

await is required when:

Calling another async def function
Using async I/O libraries (httpx.AsyncClient, asyncpg, aioredis)
await is NOT needed for:

Synchronous operations (math, string processing, list comprehensions)
Calling regular def functions
In our OllamaEmbeddingsClient, the underlying requests.post() is synchronous (blocking). We made embed_query and embed_passages async def anyway because HybridIndexingService uses await embeddings_client.embed_query(...). The function is technically async but internally synchronous — it blocks the event loop briefly. For true async, use httpx.AsyncClient instead.

**Q3. asyncio.run() error in Jupyter.**

Jupyter notebooks already run an asyncio event loop (managed by IPython/Jupyter kernel). asyncio.run() tries to create a new event loop, but Python only allows one event loop per thread. Attempting to nest them raises:

RuntimeError: asyncio.run() cannot be called from a running event loop
Solution in Jupyter: use await directly at the top level (Jupyter supports top-level await since IPython 7.0):

# Wrong in notebook:
result = asyncio.run(my_async_function())

# Correct in notebook:
result = await my_async_function()
In production Python scripts (not notebooks), asyncio.run() is correct because there's no pre-existing event loop.

**Q4. @contextmanager and yield.**

@contextmanager is a decorator that turns a generator function into a context manager — something usable with with statements.

@contextmanager
def trace_embedding(self, trace, query: str):
    span = self.tracer.start_span(name="query_embedding", input_data={"query": query})
    try:
        yield span        # ← execution pauses here when `with` body runs
    finally:
        span.end()        # ← runs when `with` body exits (even on exception)
Usage:

with rag_tracer.trace_embedding(trace, query) as span:
    embedding = await embed_query(query)
    # span is the yielded value; this is the "with" body
# After block exits, finally runs: span.end()
The yield divides setup (before) from teardown (after). Everything before yield runs on __enter__; everything after (or in finally) runs on __exit__.

---

**Q5. Annotated[T, Depends(fn)] vs T = Depends(fn).**

Both are equivalent to FastAPI, but Annotated is preferred because:

# Old style (default value approach)
async def ask_question(opensearch: OpenSearchClient = Depends(get_opensearch_client)):
    ...

# New style (Annotated)
OpenSearchDep = Annotated[OpenSearchClient, Depends(get_opensearch_client)]
async def ask_question(opensearch: OpenSearchDep):
    ...
Annotated advantages:

Creates reusable type aliases (OpenSearchDep used across many routes)
Separates the type annotation from the default value (cleaner function signatures)
Works better with static type checkers (mypy, pyright) — the type is still OpenSearchClient, not OpenSearchClient | None
FastAPI recommends this pattern since v0.95.0
Q6. How does pydantic-settings env_nested_delimiter="__" work?

With env_nested_delimiter="__", pydantic-settings maps double-underscore in env var names to nested model attributes.

class LangfuseSettings(BaseSettings):
    host: str = "http://localhost:3000"
    enabled: bool = True
    public_key: str = ""

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
Environment variable → attribute mapping:

LANGFUSE__HOST=http://langfuse-web:3000  →  settings.langfuse.host
LANGFUSE__ENABLED=true                   →  settings.langfuse.enabled
LANGFUSE__PUBLIC_KEY=pk-lf-...           →  settings.langfuse.public_key
OPENSEARCH__VECTOR_DIMENSION=768         →  settings.opensearch.vector_dimension
Single underscore (LANGFUSE_HOST) doesn't match the delimiter and won't be parsed into the nested model — it would only match a flat Settings attribute named langfuse_host (which doesn't exist), so it's silently ignored.

Single underscore (`LANGFUSE_HOST`) is NOT matched by nested delimiter — it would only be read if `LangfuseSettings` had `env_prefix="LANGFUSE_"` and a field named `host`. This was our persistent bug: `LANGFUSE_HOST` (compose.yml) was the wrong key and was silently ignored.

---

**Q7. inspect.signature() for debugging SDK mismatches.**

```python
import inspect
print(inspect.signature(opensearch_client.search_chunks_vector))
# (query_embedding: List[float], size: int = 10, categories: Optional[List[str]] = None)
```

Used throughout our debugging when we hit `TypeError: got an unexpected keyword argument`. By inspecting the actual runtime signature, we immediately saw what parameters the method accepted vs. what we were passing. More reliable than reading source code (which might be different from the installed version).

Also used `dir(obj)` to see all available methods when an `AttributeError` said a method didn't exist — `dir(Langfuse)` revealed the v3 SDK had `start_as_current_span` instead of `.trace()`.

Q8. Strategy for managing notebook kernel state across long debugging sessions?

The core problem: Jupyter cells execute independently and can be run in any order, so NameError: name 'X' is not defined is common after a kernel restart.

Strategies we learned:

1. **Always set env vars in the first cell** before any `from src...` imports
2. **Define a "session setup" cell** that recreates all shared objects (`db`, `opensearch_client`, `ollama_embeddings`, `service`) — run it whenever the kernel restarts
3. **Use upward directory search** for project root instead of hardcoded relative paths
4. **Never rely on variables from a previous cell** that you didn't explicitly run in this session
5. **Move stable code to `.py` modules** and import it — functions defined in notebook cells are lost on kernel restart

Ultimately, the right answer is: graduate stable code out of notebooks into `src/` modules as soon as it works, so sessions start cleanly with one `import` line.


"Bootstrap cell" at the top: one cell that does all setup — env vars, imports, client creation. Always run it first:
import os
os.environ["POSTGRES_DATABASE_URL"] = "postgresql://..."
from src.services.opensearch.factory import make_opensearch_client
from src.services.embeddings.ollama_client import OllamaEmbeddingsClient
opensearch_client = make_opensearch_client()
ollama_embeddings = OllamaEmbeddingsClient()
notebook_bootstrap.py module: extract the bootstrap into a file, from notebook_bootstrap import * in one line per session.

Name cells clearly: add a print at the top of each cell showing what it defines — makes it obvious which cells to re-run after a crash.

Check globals() when confused: print([k for k in globals() if not k.startswith('_')]) shows what's currently defined.

Avoid asyncio.run() in cells: use await directly. asyncio.run() in a cell will fail if the previous cell left the event loop in a bad state.

Q
---

**Q9. requests.post with stream=True and SSE consumption.**

```python
response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = json.loads(line_str[6:])  # strip "data: " prefix
            if 'chunk' in data:
                print(data['chunk'], end='', flush=True)
            if data.get('done'):
                break
```

`stream=True` tells `requests` not to download the full response body at once — it downloads in chunks as they arrive. `iter_lines()` yields each `\n`-delimited line as it comes. SSE format: each event is `data: <json>\n\n`. Consuming it line-by-line and printing immediately gives the "streaming typewriter" effect in the Gradio UI.
response.close()    # important: close the connection when done
iter_lines() yields each newline-terminated chunk. For SSE, each event ends with \n\n — iter_lines() automatically handles this and yields non-empty lines.

---

**Q10. __all__ and from . import X in __init__.py.**


__init__.py makes a directory a Python package. When you import src.routers, Python executes src/routers/__init__.py.

from . import ask, hybrid_search, ping in __init__.py imports those modules when the package is imported — making src.routers.ask accessible as a sub-module.

__all__ = ["ask", "ping", "hybrid_search"] defines what names are exported when someone does from src.routers import *. It doesn't affect explicit imports (from src.routers import ask works whether or not ask is in __all__).

Import resolution order:

sys.modules cache (already imported → return cached)
Built-in modules (math, os)
Frozen modules
Import path (sys.path) — first match wins
The bug we hit: from src.routers import agentic_ask, hybrid_search, ping in main.py but __init__.py only had from . import ask, hybrid_search, ping — agentic_ask wasn't exported from __init__.py, causing ImportError: cannot import name 'agentic_ask'.


---

## 11. System Design

**Q1. Scale to 1 million documents.**

Current system (81 chunks) → 1M documents (~10M chunks at 10 chunks/paper):

- **OpenSearch**: switch from single-node to a multi-node cluster (3+ data nodes). Use index sharding (10 primary shards). Consider dedicated coordinator nodes for query routing.
- **Embedding at scale**: batch indexing pipeline (Airflow) with parallelism. GPU cluster for embedding generation. Use async bulk indexing, not one-at-a-time.
- **kNN search at 10M vectors**: HNSW still works but needs more RAM. Consider using faiss engine (GPU-accelerated) instead of nmslib. Or switch to a dedicated vector DB (Pinecone, Weaviate, Qdrant) with distributed indexing.
- **PostgreSQL**: add read replicas for metadata queries. Consider partitioning the `papers` table by `categories` or `published_date`.
- **Redis**: switch from Basic to Standard tier with replication. Implement semantic caching (embed query, find similar cached queries) not just exact-match.
- **API**: horizontal scaling via AKS with 10+ replicas. Add a load balancer.



OpenSearch: single-node works for thousands of docs, not millions. Solution: move to a 3-node OpenSearch cluster (1 master, 2 data nodes). Shard the index (e.g., 3 primary shards). Use FAISS engine instead of nmslib for k-NN (handles larger indices better). Consider index partitioning by category or date.

Embedding: indexing 1M chunks at 100 chunks/batch = 10,000 API calls to Ollama. Solution: run multiple Ollama instances behind a load balancer, or switch to a dedicated embedding service (Jina v3 paid tier, or deploy a dedicated text-embedding-inference server on GPU). Target: 10K+ chunks/second throughput.

FastAPI: single uvicorn worker becomes a bottleneck under load. Solution: multiple workers (uvicorn --workers 4) or horizontal scaling behind a load balancer. Use async database drivers (asyncpg instead of psycopg2).

PostgreSQL: 1M rows is fine for Postgres at this schema, but add indices on arxiv_id, published_date, categories. Consider read replicas for the API if query load is high.

Redis cache: 1M possible queries means more cache misses. Implement semantic caching (cache embeddings, find semantically similar previous queries) to improve hit rate beyond exact-match.

---

**Q2. Multi-tenancy.**

Options, from simplest to most isolated:

1. **Row-level security**: add `user_id` column to OpenSearch documents and PostgreSQL tables. Filter all queries by `user_id`. Cheap but all users share infrastructure.
2. **Index-per-tenant**: each user gets their own OpenSearch index (`arxiv-papers-chunks-user123`). Better isolation, but 1000 users = 1000 indices.
3. **Cluster-per-tenant**: full infrastructure isolation. Most secure, most expensive.

For most use cases, option 1 with proper authentication middleware (JWT → user_id extraction → filter injection) is sufficient. Add API key management (user creates API keys, each request authenticated against the key).

Multi-tenancy means different users see only their own data. Two approaches:

Approach 1 — Separate OpenSearch indices per tenant: papers_tenant_123_chunks, papers_tenant_456_chunks. Complete isolation, easy data deletion per tenant. Downside: doesn't scale to thousands of tenants (too many indices).

Approach 2 — Single index with tenant_id field + filter: Add tenant_id to every document. Every search query includes filter: {term: {tenant_id: "123"}}. Cheaper (one index), scales to many tenants. Downside: one tenant can't "pollute" another's search quality but you must be careful that filters are always applied.

PostgreSQL: add tenant_id to the papers table, enforce via row-level security (Postgres RLS) or application-level filtering.

Auth: JWT tokens containing tenant_id, validated by FastAPI middleware, injected into all queries.
---

**Q3. Indexing all 33 papers with error handling.**



Steps:

Query DB for papers where raw_text IS NULL AND pdf_url IS NOT NULL
Check PDF cache (data/arxiv_pdfs/) for already-downloaded PDFs
Download missing PDFs via arxiv_client.download_pdf(paper) with exponential backoff
Parse each PDF with pdf_parser.parse_pdf(pdf_path) — Docling is slow (~30s/paper), process in batches of 5 concurrently (asyncio.gather with semaphore)
Update DB: paper.raw_text = pdf_content.raw_text; paper.sections = pdf_content.sections
Run index_papers_batch() for newly-parsed papers
Error handling: wrap each paper in try/except, log failures, continue processing. Track success/failure counts. Retry failed papers up to 3 times with exponential backoff. Return a summary dict.

Estimated time: ~30s × 30 papers ÷ 5 concurrent = ~3 minutes for parsing. Indexing 30 papers × ~25 chunks each = ~750 additional chunks to embed and index.

```python
async def index_all_papers(db, service):
    failed = []
    for paper in db.get_all_papers():
        try:
            if not paper.raw_text:
                # Re-parse PDF
                pdf_path = download_pdf(paper)
                content = await pdf_parser.parse_pdf(pdf_path)
                paper.raw_text = content.raw_text
                paper.sections = content.sections
                db.update_paper(paper)
            
            await service.index_paper(paper, replace_existing=False)
            
        except Exception as e:
            logger.error(f"Failed {paper.arxiv_id}: {e}")
            failed.append({"arxiv_id": paper.arxiv_id, "error": str(e)})
    
    return {"indexed": 33 - len(failed), "failed": failed}
```

Run as an Airflow DAG task with retries (`retries=3, retry_delay=timedelta(minutes=5)`). Skip papers already indexed (check `_count` with `arxiv_id` filter before processing).

---

**Q4. Semantic caching.**

Exact-match Redis caching (our current implementation) only hits on identical query strings. Semantic caching also matches similar queries.

Implementation:
1. When a query comes in, embed it
2. Search a "cache index" in OpenSearch (or Redis with vector support) for similar past queries (cosine similarity > 0.95)
3. If found, return the cached answer
4. If not found, run the full RAG pipeline, store result with the query embedding

The threshold (0.95) controls precision/recall of cache hits. Lower threshold = more cache hits but more wrong answers served from cache.



Current Redis cache: exact string match on query. "How does diffusion work?" and "Explain diffusion-based code repair" are treated as different queries even though they'd retrieve the same chunks.

Semantic caching approach:

Embed the incoming query: q_emb = embed_query(query) (~100ms)
Search a small secondary vector store (Redis with RedisVL, or a tiny in-memory FAISS index) for cached queries with cosine similarity > 0.95
If hit: return cached response immediately (~5ms total)
If miss: run full RAG pipeline, cache both the query embedding and the response
Implementation:

# On query
q_emb = await embed_query(query)
cached = await semantic_cache.find_similar(q_emb, threshold=0.95)
if cached:
    return cached.response

# On response
await semantic_cache.store(q_emb, response)
Cache invalidation: TTL of 24 hours (papers don't change often). Or invalidate on re-indexing events.

---

**Q5. A/B testing embedding models or chunking strategies.**

1. Create two OpenSearch indices: `arxiv-chunks-v1` (jina-v2, 600-word chunks) and `arxiv-chunks-v2` (new model, 400-word chunks)
2. Add a feature flag or user cohort assignment (50% of users → v1, 50% → v2)
3. LangFuse records which index was used per trace (in metadata)
4. Define evaluation metrics: answer quality (human ratings or LLM-as-judge), retrieval precision, latency
5. After N requests per variant, use statistical significance testing (Mann-Whitney U) to determine winner
6. Gradually shift traffic to the winner (canary deployment)

Shadow indexing: maintain two OpenSearch indices simultaneously — one with jina-v2-base-en (768-dim), one with jina-v3 (1024-dim, if API becomes available again)
Request routing: for each query, flip a coin (or use a user cohort): 50% search index A, 50% search index B
Logging: in LangFuse, tag each trace with embedding_model: "jina-v2" or embedding_model: "jina-v3" as metadata
Scoring: collect user feedback (thumbs up/down) or run automated evaluation (LLM-as-judge) against a golden question set
Analysis: compare MRR, user satisfaction scores, and latency between cohorts
Safety: don't expose the A/B split to the same user in the same session (consistent experience). Keep the test running for at least 100 queries per variant for statistical significance.

---

**Q6. Monitoring retrieval quality degradation.**

Signals to monitor:
- **Average relevance scores** of retrieved chunks over time (from search metadata in LangFuse)
- **"No relevant information found" rate** — if this increases, retrieval is failing
- **User feedback scores** (thumbs up/down) trended over time in LangFuse
- **Index staleness** — when were documents last indexed vs. when were papers published? A growing gap means users query about recent papers not yet indexed
- **Guardrail score distribution** — if average scores drop, either user queries changed or the LLM's domain classification degraded

Alerts: set up LangFuse score alerts or export metrics to Azure Monitor / Prometheus.

"No information found" response rate > 20%
Average top-1 cosine similarity < 0.80
Guardrail rejection rate doubles from baseline
Root causes of degradation: embedding model changed, new paper categories added but not indexed, index partially corrupted (check _count vs expected), drift in user query language.

---

**Q7. Papers too large for context window.**

qwen2.5:3b has a ~32K token context window. A 80K character paper = ~20K tokens — fits in one call, but sending the whole paper as context defeats the purpose of retrieval.

Strategies:
1. **Map-reduce**: split paper into sections, summarize each section independently with the LLM, then summarize the summaries. Final answer generated from section summaries.
2. **Better chunking**: our current approach (600 words) already handles this — we never pass the full paper. But if a single answer requires synthesizing information from 10+ chunks that exceed context, we need to select the most relevant subset.
3. **Hierarchical indexing**: index both individual chunks AND section-level summaries. Use section summaries for initial retrieval, then drill into chunks for the answer.
4. **Reranking**: retrieve 20 chunks, use a reranker model (cross-encoder) to score all 20 against the query, keep top 3 that fit in context.


Current approach (chunking + retrieval): already handles this — we only send the 3 most relevant chunks (~600 words each = ~900 tokens), not the whole paper. The context window is never an issue.
Map-reduce for summarization: if the task is "summarize this entire paper", split into chunks, summarize each chunk independently, then summarize the summaries.
Hierarchical retrieval: first retrieve at the document level (which papers are relevant?), then at the chunk level (which sections of those papers answer the question).
Iterative retrieval: if first answer is incomplete, retrieve more chunks and continue the conversation.
Long-context models: switch to a model with 128K context (e.g., Qwen2.5-72B-Instruct) for tasks requiring full-paper context.

Pipeline (Airflow DAG, scheduled daily at 6 AM):

@dag(schedule="0 6 * * *")
def daily_arxiv_ingestion():
    
    @task
    async def fetch_new_papers():
        # Only fetch papers published in the last 24 hours
        from_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        papers = await arxiv_client.fetch_papers(from_date=from_date, max_results=50)
        return [p for p in papers if not paper_repo.exists(p.arxiv_id)]  # skip existing

    @task
    async def process_and_store(new_papers):
        for paper in new_papers:
            pdf_path = await arxiv_client.download_pdf(paper)
            if pdf_path:
                content = await pdf_parser.parse_pdf(pdf_path)
                paper.raw_text = content.raw_text
                paper.sections = content.sections
            paper_repo.upsert(paper)
        return [p.arxiv_id for p in new_papers]

    @task
    async def index_new_papers(arxiv_ids):
        papers = paper_repo.get_by_ids(arxiv_ids)
        await indexing_service.index_papers_batch(papers, replace_existing=False)
        # replace_existing=False: skip papers already in OpenSearch

    new = fetch_new_papers()
    stored = process_and_store(new)
    index_new_papers(stored)
Idempotency: paper_repo.exists() and replace_existing=False ensure re-runs don't duplicate data.

---

Q8. If Ollama at 10.2.2.183 goes down, how would you design graceful degradation?

Current state: if 10.2.2.183:11434 is unreachable, all embedding and generation fail → every request returns 503.

Graceful degradation levels:

Level 1 — BM25 fallback for search (no embedding needed):

try:
    query_embedding = await embeddings_client.embed_query(query)
    use_hybrid = True
except Exception:
    query_embedding = None
    use_hybrid = False  # Fall back to BM25-only search
    logger.warning("Ollama unavailable, using BM25-only search")
Level 2 — Cached responses still served: Redis cache doesn't need Ollama. Exact-match cached queries still return instantly. Semantic cache also works if embeddings are cached.

Level 3 — Circuit breaker: After 5 consecutive Ollama failures, open a circuit breaker for 60 seconds — reject embedding/generation requests immediately rather than waiting for timeout. Reduces cascade latency.

Level 4 — Fallback to secondary Ollama: Configure OLLAMA_HOST_FALLBACK=http://10.2.2.184:11434. If primary fails, retry on secondary.

Level 5 — Cloud API fallback: If local Ollama is down, fall back to Ollama Cloud or a cloud LLM API (with appropriate cost controls).


**Q9. Graceful degradation when Ollama server is down.**

Current behavior: 500 Internal Server Error.

Better design:
1. **Health check at startup**: `lifespan` checks Ollama health and logs a warning but doesn't crash the API
2. **Timeout + retry**: set `httpx.Timeout(30)` on Ollama calls, retry once after 5 seconds
3. **Fallback model**: if `10.2.2.183` is unreachable, try the local `rag-ollama` container (even with no models, at least it fails fast with a clear error)
4. **Graceful error response**: return HTTP 503 with `"LLM service temporarily unavailable"` instead of 500
5. **Circuit breaker**: after 3 consecutive Ollama failures, immediately return 503 without trying Ollama (saves latency), reset after 60 seconds
6. **Monitoring alert**: LangFuse trace showing Ollama errors → trigger Azure Monitor alert → PagerDuty

---

**Q10. 1000 concurrent streaming users.**

Current architecture single points of failure and bottlenecks:

- **Ollama throughput**: one qwen2.5:3b-instruct model on one GPU can serve ~5–10 concurrent generation requests. For 1000 users: either queue requests (Redis queue + async workers), or run multiple Ollama replicas across multiple GPUs.
- **FastAPI**: single process handles async streaming well for I/O-bound work. Scale to 10+ replicas via AKS HorizontalPodAutoscaler.
- **OpenSearch**: search is fast (~10ms), handles 1000 concurrent queries easily on a single node. No bottleneck here.
- **Redis**: streaming responses bypass cache (no point caching per-token streams). Cache only the final assembled answer for subsequent identical queries.
- **LangFuse**: async trace flushing with `flush_at=15` and `flush_interval=1.0` — traces are batched and non-blocking. Should handle 1000/s without issue.

Architecture change: introduce a job queue (Redis + Celery or AsyncIO workers) so streaming generation requests are queued when Ollama is saturated, rather than all hitting Ollama simultaneously.


Current architecture single points of failure and bottlenecks:

Single FastAPI process → multiple replicas: deploy 4–8 replicas behind a load balancer (Azure Container Apps auto-scaling, or K8s HPA). Each replica can handle ~100 concurrent streaming connections.

Single Ollama instance → GPU cluster: one qwen2.5:3b-instruct generates ~50 tokens/second. 1000 concurrent users × ~300 token response = 300,000 tokens to generate. Need 6,000 tokens/second throughput → 120 Ollama instances. Use a vLLM server instead (designed for concurrent inference, batches requests automatically, much higher throughput per GPU).

OpenSearch node → cluster: search latency at 1000 RPS needs a 3-node cluster with dedicated coordinator nodes.

Redis single node → Redis Cluster: for 1000 concurrent cache lookups, Redis Cluster with 3 shards handles millions of ops/second.

Connection pooling: increase postgres_pool_size to match the number of concurrent database operations (not requests — most requests don't hit the DB).

CDN for Gradio static assets: put a CDN in front of the Gradio UI to serve static files.

Async everything: replace requests with httpx.AsyncClient in OllamaEmbeddingsClient and OllamaClient so Ollama HTTP calls don't block the FastAPI event loop.
---

*Quick Reference — Core Configs*

```
OpenSearch index:    arxiv-papers-chunks
Embedding model:     jina/jina-embeddings-v2-base-en (768-dim, on 10.2.2.183:11434)
LLM model:          qwen2.5:3b-instruct (on 10.2.2.183:11434)
Vector space:        cosinesimil (HNSW via nmslib, ef_construction=512, m=16)
Chunk size:          600 words, 100-word overlap
Postgres port:       5433 (host) → 5432 (container)
LangFuse UI:         http://localhost:3001 (host) / http://langfuse-web:3000 (internal)
Redis port:          6379 (host, no auth) / 6380 (LangFuse Redis, auth required)
```


OpenSearch index:  arxiv-papers-chunks
Embedding model:   jina/jina-embeddings-v2-base-en (768-dim, on 10.2.2.183:11434)
LLM model:         qwen2.5:3b-instruct (on 10.2.2.183:11434)
Vector space:      cosinesimil (HNSW via nmslib)
Chunk size:        600 words, 100-word overlap
Postgres port:     5433 (host) → 5432 (container)
LangFuse UI:       http://localhost:3001
OpenSearch UI:     http://localhost:5601 (OpenSearch Dashboards)
Airflow UI:        http://localhost:8080 (admin/admin)
FastAPI docs:      http://localhost:8000/docs