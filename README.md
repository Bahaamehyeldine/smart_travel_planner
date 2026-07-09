# ✈️ Smart Travel Planner

> An end-to-end AI travel recommendation system combining RAG retrieval, ML classification, and LLM generation in a production-ready full-stack architecture.

Built as a portfolio project for the SE Factory AI Engineering Bootcamp. Given a natural language query, the system retrieves semantically relevant destination information from a pgvector index, classifies the user's travel style using a trained scikit-learn classifier, and synthesizes a grounded recommendation via a Groq-hosted LLM — all orchestrated by a LangGraph stateful agent.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [ML Pipeline](#ml-pipeline)
- [RAG System](#rag-system)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Engineering Decisions](#engineering-decisions)
- [Known Limitations](#known-limitations)

---

## Architecture
┌─────────────────────────────────┐
                    │        React Frontend            │
                    │     Vite · localhost:5173        │
                    └───────────────┬─────────────────┘
                                    │ POST /api/chat
                                    ▼
                    ┌─────────────────────────────────┐
                    │       FastAPI Backend            │
                    │    uvicorn · localhost:8000      │
                    │  Pydantic validation · structlog │
                    └───────────────┬─────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │       LangGraph Agent            │
                    │                                  │
                    │  retrieve ──► classify ──► gen   │
                    └──────┬──────────┬──────────┬────┘
                           │          │          │
                ┌──────────▼──┐  ┌────▼────┐ ┌──▼──────────────┐
                │  pgvector   │  │ joblib  │ │  Groq API        │
                │  440 chunks │  │ model   │ │  Llama 3.1 8B    │
                │  cosine sim │  │ RF + GB │ │  ~500ms p50      │
                └─────────────┘  └─────────┘ └─────────────────┘
                           │
                ┌──────────▼──────────────────────┐
                │   PostgreSQL 16 + pgvector 0.8   │
                │         port 5432                │
                └──────────────────────────────────┘
**Agent execution flow:**
1. **retrieve_node** — embeds user query with `all-MiniLM-L6-v2`, queries pgvector via cosine similarity, returns top-5 Wikivoyage chunks (threshold: 0.2)
2. **classify_node** — extracts keyword features from query, runs through trained GradientBoosting pipeline, returns predicted travel style + confidence
3. **generate_node** — assembles retrieved chunks + style prediction into a structured prompt, calls Groq API, returns grounded recommendation

---

## Tech Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| Frontend | React + Vite | 18 / 5 | Fast HMR, minimal config, industry standard |
| Backend | FastAPI | 0.111 | Async-native, auto OpenAPI docs, Pydantic integration |
| Agent | LangGraph | 0.0.69 | Stateful graph execution, conditional routing, observability |
| LLM | Groq (Llama 3.1 8B) | — | Free tier, sub-second latency, OpenAI-compatible API |
| Embeddings | sentence-transformers | 2.7.0 | `all-MiniLM-L6-v2`: 384 dims, fast, strong retrieval performance |
| Vector DB | PostgreSQL + pgvector | 16 / 0.8.2 | Colocated with relational data, no additional infra |
| ML | scikit-learn | 1.4.2 | Reproducible pipelines, Pydantic-validated boundaries |
| ORM | SQLAlchemy async | 2.0.30 | Type-safe, async-first, alembic migrations |
| Validation | Pydantic | v2 | Enforced at every external boundary |
| Logging | structlog | 24.1.0 | Structured JSON logs, no print statements |
| Testing | pytest + pytest-asyncio | 8.2 | 43 tests, mocked boundaries, no live deps required |
| Containers | Docker Compose | — | Single-command full stack |

---

## Project Structure
smart_travel_planner/
├── backend/
│   ├── app/
│   │   ├── agent/
│   │   │   └── graph.py                # LangGraph StateGraph — retrieve → classify → generate
│   │   ├── api/routes/
│   │   │   ├── chat.py                 # POST /api/chat, GET /api/history
│   │   │   └── health.py               # GET /api/health
│   │   ├── core/
│   │   │   ├── config.py               # Pydantic BaseSettings, lru_cache singleton
│   │   │   └── database.py             # Async SQLAlchemy engine, AsyncSessionLocal
│   │   ├── ml/
│   │   │   ├── feature_extractor.py    # Keyword features, price tier, region encoding
│   │   │   ├── embedding_extractor.py  # sentence-transformers + PCA reduction
│   │   │   ├── wikivoyage_fetcher.py   # Rate-limited API fetcher, disk cache, retry
│   │   │   ├── train.py                # Phase 2a/2b training, GridSearchCV, results.csv
│   │   │   └── run_feature_extraction.py
│   │   ├── models/                     # SQLAlchemy ORM models
│   │   └── rag/
│   │       ├── indexer.py              # Chunk → embed → store in pgvector
│   │       └── retriever.py            # CTE-optimized cosine similarity search
│   ├── tests/
│   │   ├── conftest.py                 # Shared fixtures (client, mock_db_session)
│   │   ├── test_api.py                 # 14 endpoint tests
│   │   ├── test_feature_extractor.py   # 24 ML unit tests
│   │   └── test_retriever.py           # 5 RAG tests
│   ├── alembic/                        # Database migrations
│   ├── pytest.ini
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                     # Chat UI — messages, typing indicator, suggestions
│   │   └── App.css                     # Dark theme, responsive layout
│   ├── Dockerfile
│   ├── package.json
│   └── vite.config.js
├── data/
│   ├── processed/
│   │   ├── destinations_labeled.csv    # 200 destinations, 6 classes
│   │   ├── features.csv                # 200 × 39 feature matrix
│   │   └── results.csv                 # All experiment runs tracked
│   └── raw/
│       └── wikivoyage_cache/           # 200 cached Wikivoyage articles
├── docker-compose.yml
├── .env.example
└── README.md
---

## Quickstart

### Prerequisites

- Docker Desktop with WSL2 integration enabled
- Groq API key — free at [console.groq.com](https://console.groq.com)

### 1. Clone and configure

```bash
git clone https://github.com/Bahaamehyeldine/smart_travel_planner
cd smart_travel_planner
cp .env.example .env
# Required: add GROQ_API_KEY to .env
```

### 2. Start the full stack

```bash
docker compose up -d
```

| Service | URL |
|---------|-----|
| React frontend | http://localhost:5173 |
| FastAPI backend | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

### 3. Initialize the RAG index (first run only)

```bash
cd backend
source ../.venv/bin/activate
python -m app.rag.indexer
# ✅ RAG index built: 440 chunks in pgvector
```

### 4. Run tests

```bash
cd backend
python -m pytest tests/ -v
# 43 passed in 9.40s
```

### Local development (without Docker)

```bash
# Terminal 1 — backend
cd backend
source ../.venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm run dev
```

---

## ML Pipeline

### Dataset Construction

**200 destinations** labeled across 6 travel style classes using a systematic rubric applied to Wikivoyage "Understand" and "Do" sections:

| Class | Count | Primary Signal |
|-------|-------|---------------|
| Culture | 56 | Museum/temple/heritage activity count ≥ 3 |
| Adventure | 36 | Outdoor/extreme activity count ≥ 3 |
| Relaxation | 31 | Spa/wellness/beach characterization language |
| Budget | 27 | Hostel/street food keyword presence |
| Luxury | 25 | 5-star/exclusive/private villa keyword presence |
| Family | 25 | Kid-friendly activity and facility presence |

**Data collection:** Wikivoyage MediaWiki API with rate limiting (4–7 second random jitter), User-Agent header per MediaWiki etiquette, tenacity retry on timeout, MD5-hashed disk cache with version prefix for reproducibility.

### Phase 2a — Keyword Features (37 features)

```python
# Per-class keyword matching
f"{class}_keyword_count"    # raw keyword matches
f"{class}_keyword_binary"   # 1 if count >= CLASS_THRESHOLDS[class]
f"{class}_keyword_ratio"    # count / total_keywords

# Price tier from Sleep section
"price_tier"                # 1=Budget, 2=Mid-range, 3=Luxury

# Geographic signal
f"region_{name}"            # one-hot, 18 geographic regions
```

**Thresholds:** Activity classes (Adventure/Relaxation/Culture) require ≥ 3 matches. Keyword classes (Budget/Luxury/Family) require ≥ 1 match.

### Phase 2b — Embedding Features (87 features total)

- `all-MiniLM-L6-v2` embeddings of Wikivoyage "Understand" sections (384 dims, L2-normalized)
- PCA reduction: 384 → 50 components, fitted on training indices only (no test set leakage)
- Concatenated with Phase 2a keyword features: 37 + 50 = 87 total features

### Training Protocol

- **Split:** Stratified 80/20 train/test, `random_state=42`
- **Evaluation:** `StratifiedKFold(n_splits=5)`, macro F1 scoring
- **Class imbalance:** `class_weight='balanced'` for LR/RF; `sample_weight` for GradientBoosting
- **Tuning:** `GridSearchCV` on winning Phase 2a model
- **Experiment tracking:** all runs logged to `data/processed/results.csv`

### Results

**Phase 2a — Baseline model comparison:**

| Model | CV Accuracy | CV Macro F1 |
|-------|------------|-------------|
| LogisticRegression | 0.331 ± 0.054 | 0.318 ± 0.063 |
| **RandomForest** | **0.356 ± 0.058** | **0.333 ± 0.062** |
| GradientBoosting | 0.344 ± 0.052 | 0.320 ± 0.060 |

**Phase 2a — After tuning** (`max_depth=5, n_estimators=100`):

| | CV Macro F1 |
|--|------------|
| Baseline RandomForest | 0.333 |
| Tuned RandomForest | **0.381** (+0.047) |

**Phase 2a — Per-class test set results:**

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Adventure | 0.600 | 0.429 | 0.500 | 7 |
| Family | 0.500 | 0.400 | 0.444 | 5 |
| Budget | 0.429 | 0.500 | 0.462 | 6 |
| Culture | 0.400 | 0.364 | 0.381 | 11 |
| Luxury | 0.167 | 0.200 | 0.182 | 5 |
| Relaxation | 0.125 | 0.167 | 0.143 | 6 |

**Phase 2b — Embedding impact:**

| Phase | Model | CV Macro F1 |
|-------|-------|-------------|
| 2a tuned | RandomForest | 0.381 |
| 2b | GradientBoosting | 0.367 |

Embeddings improved Luxury F1 (+0.056) but showed instability on small classes due to ~6 zero-vector destinations with missing cache entries.

**Final model:** `data/models/GradientBoosting_phase2b_v1.joblib`

---

## RAG System

### Index Composition

| Property | Value |
|----------|-------|
| Source | 200 Wikivoyage articles |
| Chunks | 440 (section-based: Understand + Do + Sleep) |
| Embedding model | `all-MiniLM-L6-v2` (384 dims, L2-normalized) |
| Storage | pgvector `vector(384)` column |
| Distance metric | Cosine similarity (`<=>` operator) |

**Chunking strategy:** Each Wikivoyage section becomes one chunk, prefixed with destination name and section name for contextual grounding. Minimum section length: 50 characters. Falls back to `full_text` (first 5,000 chars) if no sections extracted.

### Retrieval Query

```sql
WITH distances AS (
    SELECT chunk_text, source_document, chunk_index,
           1 - (embedding <=> '[...]'::vector) AS similarity
    FROM rag_chunks
)
SELECT * FROM distances
WHERE similarity >= 0.2
ORDER BY similarity DESC
LIMIT 5;
```

CTE computes distance once per row — avoids the triple evaluation that would occur with inline expressions in WHERE, ORDER BY, and SELECT simultaneously.

### Sample Retrieval Quality

Query: "adventure activities hiking mountains"
→ Banff          similarity=0.582
→ Wadi Rum       similarity=0.475
→ Andasibe       similarity=0.475
Query: "relaxing beach spa wellness"
→ Hammamet       similarity=0.497
→ Playa del Carmen similarity=0.470
→ Bora Bora      similarity=0.454
Query: "cultural heritage temples museums history"
→ Nara           similarity=0.381
→ Kathmandu      similarity=0.377
→ Luang Prabang  similarity=0.373
---

## API Reference

### `GET /api/health`

Returns API and database connectivity status.

```json
{
  "status": "ok",
  "database": "ok",
  "version": "1.0.0"
}
```

### `POST /api/chat`

**Request body:**
```json
{
  "message": "I want adventure sports in the mountains",
  "session_id": "optional-uuid"
}
```

**Validation:** message must be 1–1000 non-whitespace characters.

**Response:**
```json
{
  "response": "Based on your adventurous spirit, I recommend...",
  "predicted_style": "Adventure",
  "style_confidence": 0.742,
  "chunks_retrieved": 5,
  "session_id": null
}
```

### `GET /api/history?limit=10`

Returns recent agent runs. `limit` is bounded to [1, 100].

**Interactive docs:** `http://localhost:8000/docs`

---

## Testing

```bash
cd backend
python -m pytest tests/ -v
# ========================= 43 passed in 9.40s =========================
```

| File | Tests | Coverage |
|------|-------|---------|
| `test_feature_extractor.py` | 24 | Keyword counting, case sensitivity, price tier extraction, binary thresholds, ratio invariants, region encoding |
| `test_api.py` | 14 | Input validation (422s), endpoint structure, agent invocation assertion, query parameter bounds |
| `test_retriever.py` | 5 | DB error graceful handling, return structure, similarity types, model singleton caching |

**Design principles:**
- All tests use mocked database and API calls — no live dependencies required
- `conftest.py` centralizes shared fixtures across test files
- `pytest.ini` sets `asyncio_mode = auto` — no per-test `@pytest.mark.asyncio` decorators needed
- Agent invocation verified with `mock_graph.ainvoke.assert_called_once()` — not just response structure

---

## Engineering Decisions

### Why LangGraph over a sequential function pipeline?

A plain `recommend(query)` function handles the happy path but breaks down when:
- Retrieval returns empty results — needs conditional routing, not a crash
- The agent needs conversation memory across turns
- You need to debug which node failed in a multi-step failure

LangGraph provides typed state, conditional edges, and a full execution trace at every node. Adding a clarification node or memory requires a new node and one edge — not a rewrite.

### Why section-based RAG chunking over sliding window?

Wikivoyage articles are semantically segmented by design. Section-based chunks:
- Preserve semantic coherence — a "Do" section chunk is entirely about activities
- Give retrieval section-level granularity — an activity query retrieves the Do section, not a diluted full-article chunk
- Are simpler and more interpretable than character-based sliding windows
- Reduce noise from mixing unrelated content (price signals with activity lists)

### Why PCA on embeddings before training?

384 embedding dimensions with 200 training samples is a textbook curse-of-dimensionality problem — more features than samples allows the classifier to fit noise. PCA to 50 components:
- Captures ~85% of embedding variance
- Reduces feature count to 87 (37 keyword + 50 PCA)
- Is fitted exclusively on training indices — test set information never touches the transformation

### Why Pydantic at every external boundary?

ML pipeline failures are often caused by silent type drift — a CSV column that shifts from `float` to `str`, or an API returning an unexpected field. Pydantic schemas at every boundary convert runtime errors from silent wrong predictions into explicit, actionable errors with field-level messages.

### Why pgvector over a dedicated vector database?

- Colocated with relational data (agent runs, users, tool calls) — single infrastructure component
- Production-ready for moderate scale (millions of vectors with HNSW indexing)
- Single Docker container for the entire data layer
- Trade-off acknowledged: dedicated solutions (Pinecone, Qdrant) offer better ANN performance at very large scale

### Why Groq (Llama 3.1 8B) over GPT-4?

- Free tier with generous rate limits — appropriate for a portfolio project
- Sub-second response times — critical for chat UX
- Sufficient quality for travel recommendation synthesis
- Architecture is LLM-agnostic — swapping providers requires changing one API call in `generate_node`

### Why structlog over Python's standard logging?

structlog outputs structured JSON key-value pairs (`destination=Queenstown`, `understand_len=1462`) rather than free-form strings. This makes logs machine-parseable, filterable by field, and directly ingestible by observability platforms — a production engineering standard.

---

## Known Limitations

### Classifier performance on characterization-based classes

Relaxation (F1=0.143), Luxury (F1=0.182), and Budget (F1=0.462) underperform relative to Adventure (F1=0.500) and Culture (F1=0.381). The root cause: Wikivoyage articles do not consistently use our keyword vocabulary to describe these travel styles. Adventure and Culture destinations explicitly list activities in "Do" sections; characterization-based classes rely on descriptive language that keyword matching misses.

**Mitigations applied:**
- Phase 2b sentence embeddings improved Luxury F1 by +0.056
- RAG retrieval compensates by grounding LLM responses in actual destination content regardless of classifier output — the product experience remains high quality even when the classifier label is wrong

**Concrete next steps:**
- Expand dataset to 500+ destinations for stable per-class metrics (current test set has only 5–7 samples per class)
- Use `all-mpnet-base-v2` (768 dims) for richer semantic representation of characterization language
- Fine-tune a text classifier directly on Wikivoyage "Understand" sections rather than extracted keyword features

### WSL2 DNS instability

Wikivoyage fetching occasionally fails with DNS resolution errors under WSL2. Per-session fix:

```bash
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'
```

### Rate limiting on bulk fetch

Fetching 200 Wikivoyage articles in sequence triggers Cloudflare rate limiting despite 4–7 second jitter. The pipeline is designed for resumability — progress is saved after each successful fetch and the pipeline can be restarted without re-fetching completed articles.

---

## Author

**Bahaa Mehye Eddin**
Data Scientist & AI Engineer · Tripoli, Lebanon
SE Factory AI Engineering Bootcamp · 2026
[GitHub](https://github.com/Bahaamehyeldine)
