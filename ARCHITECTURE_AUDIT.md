# Pravaah — Architectural Audit

**Scope:** full repository at commit `1b7eb2d` (`main`) — `backend/` (FastAPI RAG service), `frontend/Legal-Summarizer/` (React/Vite SPA, nested git repo), `Dockerfile`, and the committed vector store and corpus.
**Date:** 2026-10-02
**Method:** static review of every backend module, frontend services and components, build/deploy configuration, and direct inspection of the persisted Chroma database (`chroma.sqlite3`).

> Path note: git tracks the backend as `backend/`, but the working copy on disk is `Backend/`. On Windows this doesn't matter. On Linux clones and in the Docker build, the lowercase `backend/` is the real path. This report uses `backend/` throughout.

---

## Remediation status (updated 2026-10-02)

Sections 1–5 below describe the code as audited at `1b7eb2d`. Since then the stack has moved from Gemini + `instructor-large` to local Ollama models (`nomic-embed-text` for embeddings, `hermes3:8b` for answers and translation), and the following has changed.

**Resolved**

| Finding | Resolution |
|---------|------------|
| Shared `static/output.mp3` (privacy leak / race) | One unguessable audio file per response under `static/audio/`, deleted after 60 minutes; the old path returns 404 |
| Lawyer portal mock PDF parser | Both portals upload the real file to the backend; one shared `DocumentUpload` component |
| Print-view XSS | Print uses the React-rendered Markdown, never raw model output |
| Blocking work in `async` handlers | LLM, retrieval and translation are async; OCR runs in a threadpool, page by page, capped at 30 pages |
| No upload limits / extension-only checks | Streaming 10 MB cap (413), magic-byte validation (415), client filename discarded |
| Free-form `language`, errors re-wrapped as 500 | `Language` enum (422); typed errors: 415, 422, 503 (Ollama or OCR unavailable); generic 500 without internal detail |
| TTS or translation failure fails the request | Degrades to English text or no audio, reported in `notices` |
| 4 duplicated LLM functions and 8 duplicated handlers | One `answer()` pipeline, a prompt registry (`prompts.py`), and a table-driven router; URLs unchanged |
| CWD-relative paths, scattered config | `config.py` with absolute paths and env overrides |
| Platform-dependent chunk IDs, no content versioning of the path | Corpus-relative POSIX IDs; index rebuilt (11,326 chunks) |
| `print()` logging of prompts (PII) | `logging` with timings and sizes only; `/health` endpoint |
| CORS `*` with credentials | Localhost origins plus `CORS_ORIGINS`; no credentials |
| No tests | 39 pytest tests (API, upload validation, error mapping, CORS, chunk IDs, localization, audio, summarizer paths) |
| Summarizer used the uploaded document only as a search query (§1.4, Feature 3) | Uploads and long pasted text are summarised themselves, with audience-specific prompts; retrieval spans the whole document; documents over 12k characters are condensed part by part (up to 40k characters, with a notice beyond). No job queue yet: long documents take about 10 minutes on CPU within one request |
| No one-step local run | `run.bat` sets up Ollama models, dependencies, the index and the frontend, then starts both servers; `run.bat check` reports only |
| Dockerfile (wrong Python, unpinned, 1.9 GB venv copied, `--reload`, couldn't build from this checkout) | `backend/Dockerfile`: Python 3.13, `uv sync --frozen`, non-root, OCR packages only; verified with real OCR against host Ollama |
| Committed artifacts, broken submodule | `.pyc`, `output.mp3` and the legacy 109 MB index untracked; `.gitmodules` added; READMEs written |
| Frontend: hardcoded URLs, dead code, raw Markdown, unused deps, Vite 4 dev-server CVE | `VITE_API_BASE_URL`; about 4,600 dead lines removed; Markdown rendering; ESLint working; Vite 8; `npm audit` clean |

**Still open**

- **No auth or rate limiting.** This is fine for local single-user use, but must be added before hosting. See backlog item #5.
- **No citations or section metadata** (Feature 1), and **no retrieval or answer-quality evaluation harness**. The latter matters more now, because `hermes3:8b` sticks to the retrieved context less strictly than Gemini did.
- **Latency on CPU:** about 1.5–2 minutes per answer, and translation adds up to about 5 minutes (Kannada test: 314 s). Options are a GPU, a smaller `TRANSLATE_MODEL`, or folding translation into the generation prompt.
- **Third-party data egress:** TTS text still goes to Microsoft's edge-tts service.
- **Prompt injection** from uploaded documents is still possible, though it can no longer reach the browser as script.

---

## 1. Executive Summary & Stack Overview

### 1.1 Summary

Pravaah is a **Retrieval-Augmented Generation (RAG) assistant for Indian statutory law**. It serves two audiences: citizens, who get plain-language answers, and legal professionals, who get technical IRAC-style analysis. Each audience has two modes: summarizer and advisor. Answers can optionally be translated into five Indian languages and turned into speech.

The codebase is small: about 830 lines of Python and about 7,200 lines of JSX/CSS. It works as a hackathon-grade prototype, but it is **not production-safe** in its current form. The most urgent problems are:

| # | Finding | Severity |
|---|---------|----------|
| 1 | Every request writes TTS audio to one shared file, `static/output.mp3`, which is publicly served. Concurrent users overwrite each other, and anyone can fetch the last user's legal advice as audio. | **Critical** (privacy + correctness) |
| 2 | Lawyer-portal PDF uploads are passed to a **mock parser** that returns a hardcoded fake judgment. The backend analyses fabricated text instead of the user's document. | **Critical** (correctness) |
| 3 | All endpoints are `async def` but run blocking work: Gemini calls, CPU embeddings, Tesseract OCR, and PDF rasterisation. One OCR job freezes the whole server. | **High** |
| 4 | There is no authentication, no rate limiting, and no upload size limit. CORS is `*` with credentials enabled. The Gemini API key's spend is open to anyone. | **High** |
| 5 | The LLM output is interpolated into `document.write()` in the print view. A prompt-injected document can produce XSS on the app origin. | **High** |
| 6 | The vector store was built on Windows: chunk IDs embed `PDFS\BNS.pdf`. Re-ingesting on Linux/Docker computes `PDFS/BNS.pdf`, so the deduplication misses and all 11,529 chunks are inserted again. | **Medium** |
| 7 | There are zero automated tests. `test.py` is a manual script with a hardcoded `D:\` path. | **Medium** |

### 1.2 Technology stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| API | FastAPI 0.118 / Starlette 0.48, Uvicorn | Two `APIRouter`s, Pydantic request/response models |
| Orchestration | LangChain 0.3.27 (`langchain-core` 0.3.78) | Used only for prompt templating, the Chroma wrapper and the Gemini client; no chains or agents |
| LLM | Google Gemini `gemini-2.5-flash` (answers), `gemini-2.0-flash` (translation) | Two model versions, no central configuration |
| Embeddings | `hkunlp/instructor-large` (768-d) via `HuggingFaceEmbeddings`, on CPU | About 1.3 GB model, loaded at import time |
| Vector DB | Chroma 1.1.1, embedded/persistent (`chroma_langchain_db/`, 109 MB) | Single collection `langchain`, 11,529 chunks, **committed to git** |
| Corpus | 23 statute PDFs in `backend/PDFS/` (28 MB) | IPC, BNS, CrPC, CPC, Evidence, Contract, Companies Act, etc. |
| OCR | PyMuPDF (digital text), with a fallback to `pdf2image` + Tesseract | |
| i18n / TTS | Gemini translation + `edge-tts` (Microsoft Edge online TTS) | eng / hin / kan / tam / mal / tel |
| Frontend | React 18, Vite 4, lucide-react | No router, no state library, no TypeScript, no tests |
| Packaging | `uv` (`pyproject.toml` + `uv.lock`, Python 3.13) | **The Dockerfile ignores the lockfile** and pip-installs unpinned `>=` versions on Python **3.14** |

### 1.3 Architecture pattern

This is a **single-process, script-style monolith with a thin layered split**. There is no domain or service layer and no dependency injection. Modules talk to each other through module-level globals.

```
routers/ (HTTP, by audience)  ──►  llm.py (prompt + retrieve + generate)  ──►  multilingual.py (translate + TTS)
        │                               │
        └──► ocr_processor.py           └──► vector.py (ingestion + embeddings) ──► Chroma (on-disk)
```

The split by technical concern (OCR, vector, LLM, multilingual) is sensible. The problem is that each concern is a bag of functions with hardcoded configuration and relative paths. There are no interfaces between them.

### 1.4 Core domain model and data flow

The domain is implicit. There are no domain entities, only these concepts:

- **Corpus document**: a statute PDF, split into 800-character chunks with 80-character overlap. Each chunk ID is `"{source}:{page}:{n}"`.
- **Audience**: `human` (citizen) or `professional`.
- **Mode**: `summarizer` (similarity top-7) or `advisor` (MMR top-5 of 20).
- **Language**: `eng | hin | kan | tam | mal | tel`, a free-form string with no validation.
- **Response**: `{ text, audio_path }`.

All 8 endpoints reduce to **one pipeline**, parameterised by (audience × mode) and input type (text or file):

```
[file] ──► save to routers/uploads/{uuid}_{filename} ──► PyMuPDF text (>100 chars?) ──► else pdf2image + Tesseract
                                                                     │
[text] ──────────────────────────────────────────────────────────────┤
                                                                     ▼
                                    query = entire user text (document or question)
                                                                     ▼
                         Chroma similarity (k=7)  or  MMR (k=5, fetch_k=20)   ← embeds query on CPU
                                                                     ▼
                          PROMPT_TEMPLATE[audience][mode].format(context, question)
                                                                     ▼
                                        Gemini 2.5 Flash  .invoke()  (blocking)
                                                                     ▼
                     if lang != eng: Gemini 2.0 Flash translate  .invoke()  (blocking, 2nd LLM call)
                                                                     ▼
                               edge-tts ──► static/output.mp3  (shared, overwritten per request)
                                                                     ▼
                                     { text: translated_text, audio_path: "static/output.mp3" }
```

**Semantic mismatch in the summarizer flow.** When a user uploads *their own* document to be summarised, its full text becomes the retrieval query *and* the `{question}`. The prompt then instructs the model to answer "**only** from the CONTEXT", which means only from the retrieved statute chunks. The user's document is never what gets summarised. In addition, `instructor-large` truncates input at 512 tokens, so retrieval sees only the first page or so of a long upload.

---

## 2. Code Quality & Architectural Observations

### 2.1 Strengths

- **Clean concern boundaries at the module level.** OCR ([ocr_processor.py](backend/ocr_processor.py)), ingestion and embeddings ([vector.py](backend/vector.py)), generation ([llm.py](backend/llm.py)), and translation/TTS ([multilingual.py](backend/multilingual.py)) each live in their own module. Moving to a service layer will be straightforward.
- **Digital-first PDF extraction with an OCR fallback** ([ocr_processor.py:73-102](backend/ocr_processor.py#L73-L102)). The code tries PyMuPDF first and only rasterises and runs Tesseract when the text layer is too thin. This is the right cost/accuracy trade-off.
- **Idempotent, incremental ingestion.** Deterministic chunk IDs ([vector.py:32-51](backend/vector.py#L32-L51)) plus a check against existing IDs ([vector.py:54-76](backend/vector.py#L54-L76)) make re-running ingestion cheap, within the platform caveat in §2.3.
- **Grounded, defensive prompting.** Every prompt restricts answers to the retrieved context, tells the model to say when information is missing, and appends an audience-appropriate disclaimer. The professional advisor uses IRAC structure, and the advisors use MMR retrieval for diversity. This is thoughtful domain work.
- **Fail-fast configuration.** [app.py:17-18](backend/app.py#L17-L18) refuses to start without `GOOGLE_API_KEY`. `.env` is git-ignored and was never committed.
- **Upload hygiene basics.** Uploads are checked against an extension allow-list, saved under a UUID-prefixed name, and cleaned up in a `finally` block.
- **Typed API surface.** The Pydantic `QueryRequest` and `ResponseBody` models give automatic OpenAPI docs.
- **Frontend: the right direction already exists.** `LawyerDocumentUpload.handleGenerateAnalysis` ([LawyerDocumentUpload.jsx:1017](frontend/Legal-Summarizer/src/components/lawyer/LawyerDocumentUpload.jsx#L1017)) already collapses two near-identical API calls into one parameterised function. Context providers (`PortalContext`, `ThemeContext`) keep cross-cutting UI state out of prop chains.

### 2.2 Anti-patterns and technical debt

#### Backend

| Issue | Location | Impact |
|-------|----------|--------|
| **Four near-identical pipeline functions.** Only the prompt string and the retriever settings differ. Retrieval, formatting, logging, LLM call and TTS are copy-pasted four times. | [llm.py:27-236](backend/llm.py#L27-L236) | Any fix (async, error handling, citations) must be applied four times. Drift is guaranteed. |
| **Eight near-identical endpoints across two routers.** The upload handlers are byte-for-byte duplicates apart from the target function. Routes repeat their own prefix (`/professional/query-professional-summarizer/`). | [human_router.py](backend/routers/human_router.py), [professional_router.py](backend/routers/professional_router.py) | Same as above, plus a noisy API surface. |
| **Import-time side effects.** `llm.py` constructs a Chroma client and loads the 1.3 GB embedding model at import. `app.py` raises on a missing env var at import. | [llm.py:19-24](backend/llm.py#L19-L24), [app.py:17](backend/app.py#L17) | Slow cold start. Nothing can be unit-tested without the model download and an API key. |
| **CWD-relative paths mixed with `__file__`-relative paths.** Chroma (`"chroma_langchain_db"`), corpus (`"PDFS"`) and TTS output (`"static/output.mp3"`) are CWD-relative. `StaticFiles` and uploads are `__file__`-relative. | [vector.py:11](backend/vector.py#L11), [llm.py:20](backend/llm.py#L20), [multilingual.py:69-73](backend/multilingual.py#L69-L73), [app.py:27-32](backend/app.py#L27-L32) | Starting Uvicorn from anywhere other than `backend/` writes audio to a directory that isn't served, and creates an **empty** Chroma DB, so every answer says "not available". |
| **Divergent upload directories.** `app.py` creates `backend/uploads/`. The routers compute `BASE_DIR` from their own file and write to `backend/routers/uploads/`. The routers also define an unused `STATIC_DIRECTORY`. | [app.py:28](backend/app.py#L28), [human_router.py:12-15](backend/routers/human_router.py#L12-L15) | Dead config and confusion about where temporary files live. |
| **No configuration layer.** Model names, `k`, chunk size, the voice map, the TTS rate and CORS origins are hardcoded literals. `load_dotenv()` is called in five modules. | throughout | Changing the model or retrieval settings means a code edit and a redeploy. |
| **Stale or misleading comments.** A comment claims `gemini-2.5-flash` "is not a valid model name" while `llm.py` uses it. Two comments start with "CORRECTED:". There is a typo in a public function name (`spilt_documents`). | [multilingual.py:16-18](backend/multilingual.py#L16-L18), [vector.py:15](backend/vector.py#L15) | Low, but erodes trust in the comments. |
| **Implicit enum handled by `match` with no default.** An unknown `language` value returns `(None, None)`, and `edge_tts.Communicate(None, None)` then raises. | [multilingual.py:43-66](backend/multilingual.py#L43-L66) | Any typo in `language` (`"en"`, `"hindi"`) produces a 500 instead of a 422. |
| **Deprecated LangChain APIs.** `retriever.get_relevant_documents()` should be `.invoke()`. `langchain.prompts` and `langchain.schema` are legacy import paths. | [llm.py:159](backend/llm.py#L159), [llm.py:2](backend/llm.py#L2), [vector.py:5](backend/vector.py#L5) | These will break on the LangChain 1.x upgrade. |
| **TTS reads raw Markdown.** Prompts ask for `###`, `**` and bullet points, and the full Markdown text is sent to the TTS engine and the translator. | [multilingual.py:81-84](backend/multilingual.py#L81-L84) | The audio reads out symbols, and the translator may mangle the formatting. |
| **Audio is generated but never used.** The frontend ignores `audio_path`. `SimpleAudioPlayer` is a "coming soon" stub. | [SimpleAudioPlayer.jsx](frontend/Legal-Summarizer/src/components/SimpleAudioPlayer.jsx) | Every request pays for an external TTS round trip, and leaks the audio (see §3.2), for no user value. |

#### Frontend

| Issue | Location | Impact |
|-------|----------|--------|
| **The lawyer portal uses a mock PDF parser.** `handleFile` calls `parsePDF()`, which sleeps for 1 second and returns a hardcoded "Ram Kumar Sharma v. Union of India" judgment. `parseDOCX` is also a mock. | [LawyerDocumentUpload.jsx:975-978](frontend/Legal-Summarizer/src/components/lawyer/LawyerDocumentUpload.jsx#L975-L978), [fileParser.js:4-57](frontend/Legal-Summarizer/src/utils/fileParser.js#L4-L57) | **Every PDF a lawyer uploads is silently replaced by fake text.** |
| **Allowed file types disagree between client and server.** The citizen UI allows PDF, DOCX and TXT by MIME type. The backend allows `.pdf`, `.png`, `.jpg` and `.jpeg`. | [DocumentUpload.jsx:69-78](frontend/Legal-Summarizer/src/components/DocumentUpload.jsx#L69-L78) vs [human_router.py:23](backend/routers/human_router.py#L23) | DOCX and TXT uploads always fail with a 400. Image OCR is unreachable from the UI. |
| **Hardcoded API base URL in at least 6 places** (`http://127.0.0.1:8000`). | `api.js`, `summarization.js`, `DocumentUpload.jsx`, `LawyerDocumentUpload.jsx` | The app can't be deployed anywhere except localhost. |
| **Dead or broken service layer.** `services/api.js` calls `/human/file-summarizer/`, which doesn't exist, and is never imported. `generateMockSummary` is unreachable. | [api.js:37](frontend/Legal-Summarizer/src/services/api.js#L37), [summarization.js](frontend/Legal-Summarizer/src/services/summarization.js) | Misleading abstractions. Components bypass them and call `fetch` directly. |
| **About 900 and about 430 lines of commented-out code** at the top of the two upload components. | `LawyerDocumentUpload.jsx:1-~905`, `DocumentUpload.jsx:~540-973` | Most of each file is dead code. These are 1,238- and 973-line god components. |
| **Expected fields the backend never returns.** `result.keySections` is never sent, so the KeyHighlights panel is always empty. | `DocumentUpload.jsx`, `LawyerDocumentUpload.jsx` | Silent feature gap. |
| **Unused dependencies.** `express` and `react-markdown` are installed. The LLM's Markdown output is rendered as raw text `{summary}`. | [package.json](frontend/Legal-Summarizer/package.json), [SummaryView.jsx:143](frontend/Legal-Summarizer/src/components/SummaryView.jsx#L143) | Users see `**bold**` and `###` literally. |
| **Social share is a no-op.** It shares `window.location.href`, which carries no summary state. | [SocialShare.jsx:5-6](frontend/Legal-Summarizer/src/components/SocialShare.jsx#L5-L6) | Shared links open an empty app. |

#### Repository and build hygiene

- **Broken nested repo.** `frontend/Legal-Summarizer` is a gitlink (mode `160000`) to `github.com/Darkwizard07/Legal-Summarizer`, but there is **no `.gitmodules`**. A fresh clone of Pravaah gets an empty directory and cannot build the frontend. `frontend/UI` is a 2-byte placeholder file.
- **Committed build artifacts.** `backend/__pycache__/*.pyc` (committed before the ignore rule), `backend/static/output.mp3` (the last user's audio, 916 KB) and the full Chroma DB (109 MB) are all tracked.
- **The Dockerfile is not reproducible and is oversized.**
  - It uses `python:3.14-slim`, while the project pins 3.13.
  - It pip-installs unpinned `>=` ranges and ignores `uv.lock`. `markdown` is listed in `pyproject.toml` but is unused and absent from the Dockerfile.
  - The default `torch` wheel on Linux pulls in CUDA, adding several GB.
  - `.dockerignore` excludes `venv/` but **not `.venv/`**, so the 1.9 GB Windows virtualenv is copied into the image.
  - The CMD runs `uvicorn --reload` (dev mode) in the container.
  - The embedding model isn't pre-downloaded, so the first boot pulls about 1.3 GB.
- **Empty READMEs** (root README is 2 bytes, `backend/README.md` is 0 bytes). There are no run or ingest instructions.

### 2.3 Concurrency and data integrity

There is no relational database, so there are no transactions or row locks. The integrity concerns centre on the vector store, shared files and the event loop.

| Concern | Detail | Location |
|---------|--------|----------|
| **Shared mutable output file (race condition).** | Every request writes `static/output.mp3`. Two concurrent requests interleave writes. A client may download a half-written file or another user's audio. | [multilingual.py:69-84](backend/multilingual.py#L69-L84) |
| **Event loop blocked by synchronous work.** | `model.invoke()`, `vector_store.similarity_search_with_score()` (CPU embedding of a query that can be a whole document), `chain.invoke()` for translation, `fitz`, `convert_from_path` and `pytesseract` all run **inside `async def` handlers** on the event loop thread. A 200-page scanned PDF can block the server for minutes, and all other requests queue behind it. | [llm.py:46-56](backend/llm.py#L46-L56), [multilingual.py:35](backend/multilingual.py#L35), [human_router.py:37](backend/routers/human_router.py#L37) |
| **Sequential LLM round trips (an N+1 analogue).** | Non-English requests make two Gemini calls in series (answer, then translate), then one edge-tts call. Translation could be folded into the generation prompt, or done in parallel with TTS of the English text. | [llm.py:56-62](backend/llm.py#L56-L62) |
| **Platform-dependent chunk IDs (deduplication failure).** | The stored `source` metadata is `PDFS\2024011691-1.pdf` (verified in `chroma.sqlite3`). On Linux, `PyPDFDirectoryLoader` yields `PDFS/...`, so every ID misses and **all 11,529 chunks are inserted again as duplicates**. Retrieval then returns duplicated context. | [vector.py:32-51](backend/vector.py#L32-L51) |
| **No content versioning.** | IDs depend only on path, page and position. Replacing a PDF with an amended version keeps the stale chunks (same IDs) and never re-embeds them. A removed PDF's chunks are never deleted. | [vector.py:54-76](backend/vector.py#L54-L76) |
| **Unsafe destructive operation.** | `clear_database()` runs `shutil.rmtree` on the live persist directory while the API process holds an open SQLite handle. If this is ever run while serving, it will corrupt or crash the store. | [vector.py:79-81](backend/vector.py#L79-L81) |
| **Multiple Chroma clients and duplicate model loads.** | `llm.py` and `add_to_chroma()` each instantiate a client and load `instructor-large`. Running Uvicorn with `--workers N` loads the 1.3 GB model N times and opens N embedded-SQLite clients. Chroma's embedded mode is not designed for multi-process writers. | [llm.py:19](backend/llm.py#L19), [vector.py:55](backend/vector.py#L55) |
| **Retrieval quality.** | Scores are fetched and then discarded, with no relevance threshold, so irrelevant chunks are always injected. `instructor-large` is used without its instruction prefix, and query and passage embeddings are symmetric. Whole documents used as queries are truncated to 512 tokens. | [llm.py:46-48](backend/llm.py#L46-L48), [vector.py:22-29](backend/vector.py#L22-L29) |
| **Unbounded memory in OCR.** | `convert_from_path()` rasterises **all** pages to PIL images at 200 DPI before processing. A large scanned PDF can exhaust RAM. | [ocr_processor.py:95](backend/ocr_processor.py#L95) |
| **Index structure.** | Chroma's default HNSW index is fine at about 11.5k vectors. There is no metadata index on `source`, `act` or `section` because that metadata isn't extracted, which rules out filtered retrieval (for example, "only the BNS"). | — |

---

## 3. Resilience & Security Audit

### 3.1 Error handling and unhandled failure domains

| Pattern or gap | Location | Consequence |
|----------------|----------|-------------|
| **A catch-all swallows `HTTPException`.** The upload handlers wrap everything in `except Exception`, which also catches the `HTTPException`s they raise themselves and the 400s from the delegated query handler. All of them are re-raised as 500 with `detail="An unexpected error occurred: 500: <msg>"`. | [human_router.py:31-47](backend/routers/human_router.py#L31-L47) (×4 handlers) | Wrong status codes. Clients can't tell bad input from a server fault. |
| **Raw exception text is returned to the client.** `f"...: {e}"` and `ocr_processor` return internal messages, which can include file system paths and library errors. | same | Information disclosure. |
| **No handling around external dependencies.** Gemini (quota, safety blocks, 5xx, timeouts), edge-tts (an unofficial Microsoft endpoint that can change or rate-limit) and Tesseract/Poppler binaries have no timeouts, retries, circuit breaking or fallbacks. | [llm.py](backend/llm.py), [multilingual.py](backend/multilingual.py) | One TTS outage fails *every* request, even though the text answer was already generated. |
| **Optional steps are on the critical path.** TTS and translation failures abort the whole response instead of degrading to text-only. | [llm.py:62](backend/llm.py#L62) | Availability is capped by the least reliable dependency. |
| **Tuple-based error signalling.** `process_file_to_text` returns `(bool, str)` instead of raising typed exceptions. Callers have to remember to check the flag. | [ocr_processor.py:12](backend/ocr_processor.py#L12) | Error context is lost and handling is inconsistent. |
| **No global exception handler** and no `/health` or `/ready` endpoints. | [app.py](backend/app.py) | Orchestrators can't detect a broken vector store or a missing model. |
| **Frontend error UX.** Most calls throw `API request failed with status 500` and drop the server's `detail`. Only the lawyer path reads `detail`. | [DocumentUpload.jsx:160-161](frontend/Legal-Summarizer/src/components/DocumentUpload.jsx#L160-L161) | Users get no actionable message. |

### 3.2 Input validation, auth boundaries, sanitisation and secrets

| Area | Finding | Severity |
|------|---------|----------|
| **Authentication / authorisation** | None. All 8 endpoints and `/static/*` are public. | High |
| **Cost and abuse controls** | No rate limiting, no per-client quotas, no max query length. Each request costs 1–2 Gemini calls plus CPU embedding. | High |
| **Audio data leak** | `GET /static/output.mp3` serves the most recent user's (translated) legal advice to **anyone**. The file is also committed to git. | **Critical** |
| **CORS** | `allow_origins=["*"]` with `allow_credentials=True`. Starlette reflects the request origin in this configuration, so any site can make credentialed calls. Harmless today because there is no auth, but it becomes a CSRF vector once cookies are added. | Medium |
| **Upload size and type** | No server-side size limit. The 10 MB check is client-side only and trivially bypassed. The type check uses the extension only, with no magic-byte sniffing. There is no page-count cap before OCR, so it is open to CPU and memory DoS. PIL's decompression-bomb guard is the only protection for images. | High |
| **Filename handling** | The client-supplied `file.filename` is concatenated into the path. The UUID prefix makes traversal impractical in practice (the `"{uuid}_.."` segment doesn't exist), but the name should be discarded entirely. | Low |
| **`language` parameter** | Free-form `str`. Invalid values cause a 500 (see §2.2). It should be an `Enum` or `Literal`. | Medium |
| **Prompt injection** | Uploaded document text goes verbatim into the `{question}` slot. A document containing "ignore previous instructions…" can drop the disclaimer, fabricate law, or emit HTML or JS (see the XSS row below). | Medium |
| **XSS in the print view** | `printWindow.document.write(\`…<div>${summary}</div>…\`)` interpolates **unescaped** LLM output into a same-origin `about:blank` window. Combined with prompt injection, this is script execution on the app origin. | High |
| **Third-party data egress** | User legal documents and situations are sent to Google (Gemini) and Microsoft (edge-tts). There is no user notice and no data-processing configuration. This is a compliance concern for India's DPDP Act, 2023, given the sensitivity of legal matters. | Medium |
| **Secrets** | Good: `GOOGLE_API_KEY` comes from the environment, `.env` is git-ignored and was never committed, and `.dockerignore` excludes `.env*`. Gap: no secret manager integration, and the key is also inherited by any code that imports `llm.py`. | Low |

### 3.3 Logging, observability and debugging friction

- **`print()` is the only logging mechanism.** There is no `logging` configuration, no levels, no structured fields and no request or correlation IDs.
- **PII in logs.** The full prompt (the user's legal situation or uploaded document) and the full model response are printed on every request ([llm.py:52-60](backend/llm.py#L52-L60) ×4, and `Received query:` in every router). In any hosted environment this ends up in log retention. It is a privacy liability, and it makes the logs unreadable.
- **No metrics.** There is no latency breakdown (OCR, embedding, retrieval, LLM, translate, TTS), no Gemini token or cost accounting, and no retrieval-score telemetry.
- **No tracing** of the RAG pipeline (for example LangSmith or OpenTelemetry). When an answer is wrong, the retrieved chunks are lost. They are not returned to the client and not logged in structured form.
- **No evaluation harness.** No golden question set, no retrieval recall@k, no faithfulness checks. For a legal product, regressions in answer quality are invisible.
- **Debugging friction.** Import-time model loading makes the REPL and tests slow. CWD-relative paths cause "works on my machine" failures. `--reload` inside Docker hides startup errors behind restarts.

---

## 4. High-Priority Refactoring Recommendations

### 4.1 Prioritised backlog

Ordered by **impact ÷ effort**. Items 1–3 are detailed in §4.2.

| # | Recommendation | Impact | Effort |
|---|----------------|--------|--------|
| 1 | **Collapse into a single parameterised RAG pipeline** with a central `Settings` object and absolute paths | High | Medium |
| 2 | **Make the request path non-blocking and request-isolated**: per-request audio artifacts, async I/O, upload limits, graceful degradation | High | Low–Med |
| 3 | **Fix frontend correctness and security**: remove the mock parsers, add a single API client with an env-based URL, fix the XSS, render Markdown, play audio | High | Low–Med |
| 4 | Add an error taxonomy and a global exception handler. Return correct 4xx/5xx codes and never echo raw exceptions. | High | Low |
| 5 | Basic edge security: API key or JWT auth, rate limiting (`slowapi`), an explicit CORS allow-list, server-side size and page-count limits, `language` as an `Enum` | High | Low–Med |
| 6 | Structured logging (`logging` + JSON formatter + request ID middleware). Remove prompt and response `print`s. Add `/health` and `/ready`. | Medium | Low |
| 7 | Make ingestion a separate offline CLI or job with POSIX-normalised, content-hashed chunk IDs. Stop committing the DB, and build it in CI or on first boot. | Medium | Medium |
| 8 | Test suite: unit tests for `calculate_chunk_ids`, prompt registry and OCR routing; API tests with FastAPI `TestClient` and fake LLM/vector store injected via `Depends`; a small golden-set retrieval eval | Medium | Medium |
| 9 | Docker: multi-stage build with `uv sync --frozen`, Python 3.13, CPU-only torch index, `.venv` in `.dockerignore`, no `--reload`, pre-bake the embedding model, non-root user | Medium | Low |
| 10 | Repo hygiene: register the frontend as a proper submodule (`.gitmodules`) or vendor it; delete `frontend/UI`, the tracked `.pyc` files and `output.mp3`; write READMEs | Low | Low |
| 11 | Replace deprecated LangChain APIs (`.invoke()`, `langchain_core.prompts`, `langchain_core.documents`). Unify the Gemini model version. | Low | Low |

### 4.2 Top 3 in detail

#### #1: A single parameterised RAG pipeline with central configuration (High impact / Medium effort)

**Files affected**

- [backend/llm.py](backend/llm.py): replace the four functions
- [backend/routers/human_router.py](backend/routers/human_router.py), [backend/routers/professional_router.py](backend/routers/professional_router.py): collapse into one router
- [backend/models.py](backend/models.py): add enums and a richer response
- [backend/vector.py](backend/vector.py), [backend/multilingual.py](backend/multilingual.py), [backend/app.py](backend/app.py): use the shared settings and paths
- **New:** `backend/core/config.py`, `backend/services/rag.py`, `backend/services/prompts.py`, `backend/api/deps.py`

**Strategy**

1. **`core/config.py`** holds a `pydantic-settings` `Settings` class. It defines `BASE_DIR = Path(__file__).resolve().parents[1]` and absolute `CHROMA_DIR`, `CORPUS_DIR`, `AUDIO_DIR` and `UPLOAD_DIR`. It also holds `LLM_MODEL`, `TRANSLATE_MODEL`, `EMBED_MODEL`, `CORS_ORIGINS`, `MAX_UPLOAD_MB` and `MAX_PDF_PAGES`. Delete every scattered `load_dotenv()` call. This removes the CWD dependence and the duplicate upload directories.
2. **`services/prompts.py`** holds a registry keyed by `(Audience, Mode)`:
   ```python
   @dataclass(frozen=True)
   class PromptProfile:
       template: str
       search_type: Literal["similarity", "mmr"]
       search_kwargs: dict

   PROFILES: dict[tuple[Audience, Mode], PromptProfile] = {
       (Audience.citizen, Mode.summarize):      PromptProfile(CITIZEN_SUMMARY, "similarity", {"k": 7}),
       (Audience.citizen, Mode.advise):         PromptProfile(CITIZEN_ADVICE,  "mmr", {"k": 5, "fetch_k": 20}),
       (Audience.professional, Mode.summarize): ...,
       (Audience.professional, Mode.advise):    ...,
   }
   ```
3. **`services/rag.py`** is one `RagService` class. The vector store, LLM and TTS are injected through its constructor, and it exposes `async def answer(profile, text, lang) -> RagResult`. It retrieves with `asyncio.to_thread` or `aretriever.ainvoke`, generates with `await llm.ainvoke`, and returns `text`, `sources` (chunk metadata and scores) and an optional `audio_url`. It is built once in an app **lifespan** handler and provided with `Depends(get_rag_service)`, which replaces the import-time globals and makes the service mockable in tests.
4. **One router** with two endpoints replaces the current eight:
   ```
   POST /api/v1/{audience}/{mode}          JSON  { query, language }
   POST /api/v1/{audience}/{mode}/upload   multipart file + language
   ```
   `audience` and `mode` are path `Enum`s, so FastAPI validates them. Keep the old routes as thin deprecated aliases for one release so the frontend can migrate.
5. **`models.py`**: `Language(str, Enum)`, `Audience`, `Mode`, `QueryRequest(query: constr(min_length=1, max_length=20_000), language: Language = Language.eng)`, `ResponseBody(text, sources: list[SourceRef], audio_url: str | None)`.

*Payoff:* about 350 duplicated lines shrink to about 120. Every later fix (async, citations, caching, streaming) is made in one place.

#### #2: A non-blocking, request-isolated request path (High impact / Low–Medium effort)

**Files affected:** [backend/multilingual.py](backend/multilingual.py), [backend/ocr_processor.py](backend/ocr_processor.py), [backend/llm.py](backend/llm.py) (or `services/rag.py` after #1), both routers, [backend/app.py](backend/app.py)

**Strategy**

1. **Per-request audio artifacts.** `generate_tts` writes to `AUDIO_DIR / f"{uuid4().hex}.mp3"` and returns `/static/audio/{id}.mp3`. Add a TTL cleanup (a startup background task that deletes files older than N minutes). Better still, stream the bytes from a dedicated `GET /api/v1/audio/{id}` with an unguessable ID, or skip TTS unless `include_audio=true`. **Delete `backend/static/output.mp3` from git and the working tree.**
2. **Unblock the event loop.**
   - LLM: `await model.ainvoke(prompt)`, and use `chain.ainvoke` for translation.
   - Retrieval: `await asyncio.to_thread(vector_store.similarity_search_with_score, ...)`, or the retriever's `ainvoke`.
   - OCR: `await run_in_threadpool(process_file_to_text, path)`. Better, offload to a `ProcessPoolExecutor` because Tesseract is CPU-bound.
   - Process PDFs page by page with `convert_from_path(first_page=i, last_page=i)` to bound memory, and enforce `MAX_PDF_PAGES`.
3. **Stream uploads with a size cap.** Read `file.file` in chunks and abort with a 413 when `MAX_UPLOAD_MB` is exceeded. Store to `tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=ext)` and ignore the client filename. Validate magic bytes (`%PDF-`, PNG and JPEG signatures).
4. **Graceful degradation.** Wrap translation and TTS in `try/except` with timeouts (`asyncio.wait_for`). On failure, return the English text with `audio_url=None` and a `warnings` field, not a 500. Run TTS concurrently with any remaining work (`asyncio.gather`).
5. **Fix exception flow in the routers.** Catch only domain exceptions (`UnsupportedFileError → 415`, `ExtractionError → 422`, `UpstreamLLMError → 502`) and let `HTTPException` propagate. Register a global handler that logs with the request ID and returns a generic message.
6. **Fold translation into generation where possible.** Add `"Respond in {language}."` to the prompt, which saves a full LLM round trip. Keep the separate translator only when you need the English text as well.

*Payoff:* concurrent users no longer serialise behind each other or see each other's audio, the critical privacy leak is closed, and p95 latency drops by one LLM round trip for non-English requests.

#### #3: Frontend correctness and security (High impact / Low–Medium effort)

**Files affected**

- [frontend/Legal-Summarizer/src/components/lawyer/LawyerDocumentUpload.jsx](frontend/Legal-Summarizer/src/components/lawyer/LawyerDocumentUpload.jsx)
- [frontend/Legal-Summarizer/src/components/DocumentUpload.jsx](frontend/Legal-Summarizer/src/components/DocumentUpload.jsx)
- [frontend/Legal-Summarizer/src/utils/fileParser.js](frontend/Legal-Summarizer/src/utils/fileParser.js)
- [frontend/Legal-Summarizer/src/services/api.js](frontend/Legal-Summarizer/src/services/api.js), [frontend/Legal-Summarizer/src/services/summarization.js](frontend/Legal-Summarizer/src/services/summarization.js)
- [frontend/Legal-Summarizer/src/components/SummaryView.jsx](frontend/Legal-Summarizer/src/components/SummaryView.jsx), [frontend/Legal-Summarizer/src/components/lawyer/LawyerSummaryView.jsx](frontend/Legal-Summarizer/src/components/lawyer/LawyerSummaryView.jsx)
- [frontend/Legal-Summarizer/src/components/SimpleAudioPlayer.jsx](frontend/Legal-Summarizer/src/components/SimpleAudioPlayer.jsx)

**Strategy**

1. **Remove the mock parsers.** In `LawyerDocumentUpload.handleFile`, stop calling `parsePDF` and `parseDOCX`. Keep the `File` object and send it to the backend `/upload` endpoint, the same flow the citizen portal already uses in `handleFileUploadRequest`. Delete the mock `parsePDF` and `parseDOCX` from `fileParser.js`, keeping only `loadSampleDocument` and `extractKeySections` if they are still wanted. Align accepted types with the backend: either add DOCX/TXT support server-side (`python-docx`, plain read) or restrict the UI to PDF/PNG/JPG.
2. **One API client.** Rewrite `services/api.js` as the only network module. It sets `const API_BASE_URL = import.meta.env.VITE_API_BASE_URL`, and exports `analyzeText({audience, mode, text, language})` and `analyzeFile({audience, mode, file, language})` that match the unified backend routes from #1. They should surface `detail` from error responses. Replace every inline `fetch` in the two upload components with these calls, and delete `summarization.js`.
3. **Shared upload logic.** Extract the shared drag-drop, file-validation, language-picker and submit logic into a `useDocumentAnalysis(audience)` hook plus a `<LanguageSelect>` component. The two upload components then differ only in copy and styling. Delete the roughly 1,300 lines of commented-out code.
4. **Fix the XSS.** In both `handlePrint` implementations, stop interpolating into `document.write`. Either render a print-only React subtree and call `window.print()` with `@media print` CSS, or set `textContent` on a created node. Render the summary with the already-installed `react-markdown` (it escapes raw HTML by default, so do not enable `rehype-raw`).
5. **Use the backend features you already pay for.** Wire `SimpleAudioPlayer` to `audio_url` from the response (`<audio src>`), and populate KeyHighlights from the new `sources` field (see §5, Feature 1). Remove the unused `express` dependency.

*Payoff:* the lawyer portal stops analysing fabricated documents, the app becomes deployable beyond localhost, a script-injection path is closed, and roughly 1,300 lines of dead code are removed.

---

## 5. Feature Extension & Scalability Opportunities

### Feature 1: Verifiable citations ("show me the section")

Return the retrieved chunks with their act, section and page, and render them as clickable citations next to the answer. This fills the KeyHighlights panel that already exists but is always empty. For a legal product, traceability is the main trust feature, and the data is already retrieved and thrown away in [llm.py:48](backend/llm.py#L48).

**Prerequisites**

- Refactor #1, so that `RagResult.sources` and `ResponseBody.sources` exist as an API schema change, and the API is versioned under `/api/v1`.
- **Ingestion metadata enrichment:** normalise `source` to an act name and year, and extract `section_number` and `section_title` with regex over the chunk headers. This requires a **re-index**, best done together with the chunk-ID fix in backlog item #7.
- Apply a relevance-score threshold so that weak matches aren't cited.

### Feature 2: IPC ⇄ BNS (and CrPC ⇄ BNSS) cross-reference lookup

The corpus already contains both the IPC and the BNS, and the citizen-advisor prompt already asks for "old law and new law". A deterministic mapping table, served as a dedicated endpoint and also injected into the prompt context, gives accurate, non-hallucinated section equivalence. Practitioners need this during the transition to the new criminal codes.

**Prerequisites**

- A small **relational store** (SQLite or Postgres via SQLAlchemy + Alembic migrations) for the mapping table, seeded from the official correspondence tables.
- **Metadata-filtered retrieval:** requires the `act` and `section` metadata from Feature 1, so that "BNS 303" can be fetched exactly rather than semantically.
- A hybrid retrieval layer: an exact section lookup first, then a vector search. This is a natural extension of the `RagService` from refactor #1.

### Feature 3: True "summarise my document" with long-document support

Make the summarizer actually summarise the uploaded judgment or contract: run a map-reduce summary over the user's document, then retrieve the statutes it references, then produce an annotated summary. Today the document is only used as a (truncated) search query (§1.4).

**Prerequisites**

- A **background job queue** (ARQ, RQ or Celery with Redis). Large OCR plus multi-call summarisation exceeds sensible HTTP timeouts. Expose `POST /jobs` returning a `job_id`, then `GET /jobs/{id}`, or push progress over SSE.
- **Object storage** (S3, GCS or MinIO) for uploaded files and results, with a retention policy. This replaces local temp directories, which don't survive multi-instance deployments.
- Token budgeting and chunking utilities. Separate prompt profiles for "document summary" and "statute-grounded analysis".

### Feature 4: Conversational follow-ups with session history

Allow "what if the amount was under ₹50,000?" style follow-ups that keep earlier context. Advisor use cases are naturally multi-turn.

**Prerequisites**

- **Identity:** anonymous session tokens at minimum, or user accounts (OAuth or JWT). This depends on backlog item #5 and on tightening CORS.
- **Persistence:** `sessions` and `messages` tables (Postgres plus Alembic), with **encryption at rest and a TTL or deletion policy**, because conversations contain sensitive legal facts (DPDP Act).
- A query-rewriting step (condense history into a standalone question before retrieval) added to the pipeline.
- **Streaming responses (SSE)** via `llm.astream()`. This needs the async refactor (#2) first.

### Feature 5: Corpus management and amendment tracking (admin)

An authenticated admin flow to add new acts or amendments, re-index incrementally, and mark which version of a statute an answer was based on ("as of 2026-07-01").

**Prerequisites**

- **Ingestion moved out of the API process** into a CLI or worker (backlog item #7). Use **content-hash chunk IDs** (`sha256(normalised_path + page + text)`) so that changed pages are re-embedded and orphaned chunks are deleted.
- **Collection versioning / blue-green indexes:** build `statutes_v{n}` alongside the live index and switch an alias atomically. This replaces the unsafe `rmtree` in `clear_database()`.
- **A client/server vector DB** (Chroma server mode, Qdrant or pgvector) so that multiple API replicas and the ingestion worker share one index safely.
- Role-based auth (admin vs public) and an audit log of corpus changes.

### Cross-cutting scalability enablers

| Enabler | Unlocks |
|---------|---------|
| **Response and translation cache** (Redis, keyed on `hash(profile, normalised_query, lang)`) | Lower Gemini cost and latency for common questions. Pairs with rate limiting. |
| **GPU or hosted embedding endpoint**, or a lighter model (e.g. `bge-small`, `multilingual-e5`) | Removes the 1.3 GB per-worker CPU model. Also enables multilingual queries without translating first. |
| **Stateless API replicas** (vector DB server + object storage + Redis) | Horizontal scaling behind a load balancer. |
| **RAG evaluation harness** (golden Q/A set, recall@k, faithfulness scoring in CI) | Safe iteration on prompts, chunking and models without silent quality regressions. |
