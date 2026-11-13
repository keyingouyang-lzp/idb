# iDB

**Schema-on-read structured querying over heterogeneous documents.**

iDB ingests PDFs, scanned contracts, Word files, spreadsheets and slides, and
answers natural-language questions with structured records: supplier names,
amounts, aggregated totals, each with a source citation and a confidence score.
No schema is defined up front — the output schema is inferred at query time.

```
documents ──▶ parse (OCR when needed) ──▶ chunks ──▶ dual index
                                                      │
query ──▶ vector search ──┐                          │
      └─▶ trigram FTS5 ───┼──▶ RRF fusion ──▶ diversify / coverage ──▶ LLM
                          │                                           │
                          └───────────────────────────────────────────┴──▶ records + citations
```

## What it does

- **Any common office document.** PDF (with automatic Tesseract OCR fallback),
  DOCX, XLSX, PPTX, TXT, CSV.
- **Chinese-first retrieval.** SQLite FTS5 is built with the `trigram`
  tokenizer, because the default `unicode61` tokenizer treats a Chinese
  sentence as one unbroken token and retrieves nothing.
- **Hybrid ranking.** Vector similarity (LanceDB) and trigram keyword search are
  fused with Reciprocal Rank Fusion, then diversified per file, with a
  coverage step that keeps one chunk from every candidate file.
- **Answers, not passages.** The LLM infers which fields to extract, returns a
  JSON table, and attaches the supporting quote and a confidence score.
- **Exact arithmetic over tables.** Aggregate questions ("total value of 2022
  contracts") are answered by parsing the amount column and summing it, not by
  asking the model to add rows.
- **CLI and web UI.** `idb ask` for scripts, a FastAPI backend plus a browser UI
  for day-to-day use.

## Quickstart

Requires Python 3.10+ and, for scanned PDFs, `tesseract` and `poppler`
(`brew install tesseract poppler` on macOS).

```bash
git clone https://github.com/<your-account>/idb.git
cd idb
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example ~/.idb/.env     # then fill in the API keys
idb init
idb ingest ./my-documents
idb ask "口腔CT的供应商和合同金额"
idb web                         # http://127.0.0.1:8720
```

`idb ingest` accepts a file or a directory (`--recursive` is on by default),
`--dry-run` lists what would be ingested, and `--force` re-ingests unchanged
files. `idb ask` accepts `--top`, `--json` and repeated `-f field` flags.

If the embedding endpoint is unavailable, iDB silently falls back to the local
`BAAI/bge-small-zh-v1.5` model (512-d). Retrieval then still works offline; only
extraction needs a network LLM.

## Configuration

Settings are read from `~/.idb/.env` or `./.env`.

| Variable | Default | Purpose |
|---|---|---|
| `IDB_DB_PATH` | `~/idb_data` | SQLite metadata + LanceDB vectors |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | – | OpenAI-compatible embedding endpoint |
| `IDB_EMBEDDING_MODEL` | `text-embedding-3-small` | falls back to local BGE-small-zh |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | – | Anthropic-compatible LLM endpoint |
| `IDB_LLM_MODEL` | `claude-sonnet-4-6` | model used for extraction |
| `IDB_LLM_MAX_TOKENS` | `16384` | response budget (see below) |
| `IDB_FINANCIAL_BOOST` | `0` | optional keyword boost, off by default |

## Web API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/stats` | indexed files and chunk counts |
| `POST` | `/api/ingest` | upload one file |
| `POST` | `/api/ask` | retrieve + extract |
| `GET` | `/api/search_simple?q=` | retrieval only |
| `DELETE` | `/api/files/{name}` | remove a file |
| `POST` | `/api/export` | export a result table |

## Design notes from the evaluation

Three behaviours are deliberate, and each came out of measuring the pipeline
rather than assuming it:

1. **The metadata path only participates when a filter is supplied.** Querying
   it with an empty dict degrades to `WHERE 1=1 LIMIT 30`, which returns the
   same chunks for every query and acts as a constant prior. Removing it raised
   MRR@5 from 0.600 to 0.857 at constant recall.
2. **The response budget is 16k, not 4k.** Reasoning models spend part of the
   budget on a thinking block; at 4096 tokens one call in four returned no text
   at all.
3. **Financial keyword boosting is off by default.** The increment (+0.008) is
   about 26x the spacing between adjacent RRF ranks (~0.0003), so it reorders
   the window far more aggressively than intended. On the reference corpus it
   lowered MRR@5 from 0.857 to 0.762 without improving end-to-end success.
   Set `IDB_FINANCIAL_BOOST=1` to try it on your own data.

## Reference evaluation

Measured on a private corpus of 17 Chinese hospital procurement documents
(99 chunks, 108 OCR pages, five formats); the corpus itself is not distributed.

| Configuration | Recall@20 | MRR@5 |
|---|---|---|
| Vector only | 0.857 | 0.857 |
| FTS5, default tokenizer | 0.000 | 0.000 |
| FTS5, trigram tokenizer | 1.000 | 0.886 |
| RRF fusion | 1.000 | 0.857 |
| **Full pipeline** | **1.000** | **0.857** |

End-to-end, the full pipeline answered 7/7 queries correctly, against 5/7 for
vector-only retrieval with a generic extractor, 5/7 for trigram search, and 0/7
for the default FTS5 tokenizer. Ingestion of the 108 OCR pages took about ten
minutes; query latency averaged 44 s, of which roughly 83% is LLM inference.

These numbers come from a small, single-domain corpus and are point estimates.
Treat them as a sanity check on the design, not as a benchmark.

## Limitations

- Seven queries on 17 documents cannot establish variance; the metrics above
  are single-run point estimates.
- File diversification and the coverage guarantee changed no metric at this
  corpus size — they are hypotheses about larger collections, not demonstrated
  wins.
- Table-structured aggregation works by parsing the amount column; it does not
  generalise to free-form numeric reasoning across arbitrary documents.
- Extraction quality is bounded by the LLM, and every retrieval result sent to
  a remote model leaves the machine. Embeddings can run locally; extraction
  currently cannot.

## Project layout

```
src/idb/
  cli.py                 command line: init / ingest / ask / stats / web
  config.py              environment-driven configuration
  ingestion/parser.py    format dispatch + Tesseract OCR fallback
  ingestion/chunker.py   paragraph- and table-aware chunking
  ingestion/embedder.py  batched embeddings
  query/retriever.py     hybrid retrieval, RRF, diversification, coverage
  query/extractor.py     field inference, extraction, table aggregation
  storage/lancedb_store.py   vector index
  storage/sqlite_store.py    metadata + FTS5 (trigram)
  web/server.py          FastAPI backend
  web/static/index.html  single-page UI
tests/fixtures/          synthetic documents used by the tests
```

## License

MIT. See [LICENSE](LICENSE).
