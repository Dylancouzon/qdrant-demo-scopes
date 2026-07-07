# Demo 6: Visual Docs Search

**Build brief. Self-contained: everything you need is in this file. Read it fully before writing any code.**

## What you are building

Search over 10–20k scanned pages (typewritten memos, forms, stamps, cursive
notes) **as page images, with no OCR in the primary retrieval path**, using
ColPali-style multivectors with MaxSim in Qdrant. Binary quantization cuts
vector memory to roughly 1/32 so late interaction stays affordable at corpus
scale. Results explain themselves: a patch-level heatmap lights up exactly
which regions of the page matched each query token, while an OCR pipeline
running beside it returns garbage on the same handwriting.

**The hero moment (this must work flawlessly):** a query targeting a
handwritten note lights up the exact ink strokes on the scanned page as a
heatmap, while the OCR baseline beside it returns nothing useful, with its
garbled transcript on display as proof.

**Why this demo exists:** native multivector MaxSim, quantization, and tiered
on-disk storage in one engine are what make late interaction viable at scale.
Without quantization, ColPali-scale multivectors exhaust RAM fast; engines
without native multivectors need flattening hacks that lose MaxSim semantics.

**Deployment note:** this demo runs **locally** (the page-image encoder needs
local GPU/MPS compute). That is a deliberate exception to the team's
Vercel rule. The repo still publishes to `qdrant-labs`, and the README
documents a hosted-encoder + Vercel path as future work.

**Acceptance bar:** runs end to end on a single laptop with `make run`, repo
published to `qdrant-labs`, all evals passing, hero moment reproducible.

## How to work (instructions for the building agent)

1. Work through the phases in order; pass each gate before continuing.
2. Verify Qdrant syntax against the linked docs before coding each query.
   Key references:
   - Multivectors + MaxSim: https://qdrant.tech/documentation/concepts/vectors/
   - Quantization + rescoring: https://qdrant.tech/documentation/guides/quantization/
   - Qdrant's optimized ColPali pattern (pooled prefetch + multivector
     rescore): search qdrant.tech for the "PDF retrieval at scale" tutorial and
     check github.com/qdrant/demo-colpali-optimized. If both exist, mirror
     their storage layout; if not, implement the pattern as specified in this
     file.
3. Model note: prefer the strongest **permissively licensed** ColPali-family
   checkpoint. Start from `vidore/colqwen2-v1.0` (colpali-engine); check the
   ViDoRe leaderboard for a newer permissive option before committing. Record
   the license check in the README.
4. Use executive judgment. This brief exists to produce a great demo that
   showcases Qdrant, not to enforce every literal implementation detail. If a
   detail blocks stage impact, make the smallest sensible decision, document it
   in the README, and preserve the Qdrant story.
5. Keep going until the project is complete. Build through all phases, run the
   gates and evals, deploy or publish as far as access allows, and do not stop at
   a prototype unless an external access blocker makes completion impossible.
6. If an access item under Prerequisites is missing, stop and ask Dylan
   (dylan.couzon@qdrant.com) before building around the gap.

## Prerequisites

- An Apple Silicon Mac (MPS) or NVIDIA GPU machine: ingest embeds 10–20k pages
  (hours), and every query embeds live.
- Docker (local Qdrant).
- GitHub write access to `qdrant-labs`.
- Hugging Face account (free): the model weights and datasets are public;
  a token only matters if you hit anonymous rate limits or a gated fallback.

## Locked architecture (do not relitigate these)

| Decision | Choice | Why |
|---|---|---|
| Deployment | **Fully local**: Next.js UI (dev server) + a Python FastAPI service (encoder, search, heatmaps) + Qdrant in Docker | Encoder can't live on Vercel; local Qdrant makes the whole demo work offline at a booth |
| Corpus | **Real archive scans**, DocVQA document images (~12.7k single pages from the UCSF Industry Documents Library: memos, letters, forms, handwriting, stamps). Fallback if access is a problem: a 10–20k sample of RVL-CDIP `letter`/`form`/`handwritten` classes | Real cursive and stamps make the hero moment honest. Narrative is "scanned records archive", NOT insurance claims: do not pretend the documents are something they aren't |
| Model | ColQwen-family via `colpali-engine`, 128-d patch vectors, run on MPS/CUDA for both ingest and query | One model, both sides; no drift between index and queries |
| Storage layout | Per page, named vectors: `pooled` (mean over all patch vectors, 128-d, HNSW **enabled**) and `patches` (full multivector, MaxSim comparator, HNSW **disabled** via `m: 0`, **binary quantization first**, originals on disk) | HNSW over ~1000-vector multivectors is prohibitive; the pooled vector does fast first-stage retrieval, the multivector does exact MaxSim rescoring |
| Query flow | Embed query (~20–40 token vectors) → prefetch top 100 on `pooled` (query = mean of token vectors) → rescore those 100 with full MaxSim on `patches` → top 10 | The two-stage pattern is the demo's scaling story |
| Memory meter | Computed from **live collection counts** and vector configs: full-precision bytes vs binary-quantized bytes, labeled "computed from live collection stats" | Honest and reproducible; don't screenshot-fake RAM graphs |
| OCR baseline | pytesseract on every page at ingest → chunks → BM25 (`rank_bm25`) served by the same FastAPI service, side by side in the UI | ~100 lines; fails visibly on handwriting, which is the point |
| Heatmaps | `colpali-engine`'s interpretability utilities (query-token → patch similarity maps) rendered as canvas overlays; hovering a query token switches the map | This is the "results explain themselves" feature |

**Memory math to display (recompute from real counts at runtime):** ~12.7k
pages × ~750 patches × 128-d float32 ≈ 4.9 GB full precision vs ≈ 153 MB
binary-quantized. That ~1/32 target, with rescoring keeping accuracy, is the
headline if the quantization-fidelity eval passes. If binary quantization misses
the top-5 overlap gate on 128-d patch vectors, switch to the smallest Qdrant
quantization mode that passes the gate and report the measured footprint
honestly.

## Qdrant design

One collection:

```python
vectors_config = {
  "pooled":  VectorParams(size=128, distance=COSINE),
  "patches": VectorParams(size=128, distance=COSINE,
      multivector_config=MultiVectorConfig(comparator=MAX_SIM),
      hnsw_config=HnswConfigDiff(m=0),
      quantization_config=BinaryQuantization(always_ram=True),
      on_disk=True),
}
```

(Verify exact parameter names against current client docs before coding.)
Payload: `page_id`, `source`, `doc_type`, `thumb_path`, `ocr_text` (for the
baseline and for honesty in the UI: show what OCR thought the page said).
Rescoring uses oversampling with originals on disk; verify the
`quantization` search-params syntax in the quantization guide.

Canonical search shape:

```json
{
  "prefetch": {
    "query": "<mean_query_vector>",
    "using": "pooled",
    "limit": 100
  },
  "query": "<query_token_vectors>",
  "using": "patches",
  "params": {
    "quantization": {
      "rescore": true,
      "oversampling": 2.0
    }
  },
  "limit": 10
}
```

The API route returns both Qdrant timings and model timings: encode, pooled
prefetch, MaxSim rescore, heatmap render. The UI labels those timings separately
so the audience can see which part Qdrant accelerates.

## UI

- **One search box**, dark editorial design (it will be projected).
- Results: page thumbnails with score; click → full-page view with the
  **token→patch heatmap** overlay and a token strip to switch maps.
- **Split view toggle:** same query against the OCR+BM25 baseline, side by
  side. Show the OCR transcript for the hero page so the audience sees *why*
  it failed (garbled cursive).
- **Memory meter:** persistent footer strip: full-precision vs quantized
  footprint computed live, point + patch counts, and per-query latency split
  (encode / prefetch / rescore).

## Build phases and gates

- **Phase 0: environment.** Model loads on MPS/CUDA, encodes one page + one
  query; Qdrant up in Docker; collection created. Gate: a 10-page
  hand-ingested MaxSim query returns the right page.
- **Phase 1: ingest.** Batch pipeline: render page, patch embeddings,
  pooled vector + pytesseract text → upsert; checkpointed, `--limit` flag.
  Gate: 500-page smoke corpus, two-stage query < 2.5 s end to end.
- **Phase 2: full corpus.** 10–20k pages (expect hours; run overnight if
  needed). Gate: counts match, memory meter numbers reproduce the math above,
  p95 two-stage query < 2.5 s including encoding.
- **Phase 3: heatmaps.** Gate: for 10 spot-check queries the highlighted
  patches visually correspond to the matching content (a human look, not a
  metric).
- **Phase 4: OCR baseline + split view.** Gate: at least 5 curated queries
  where ColPali retrieval succeeds and the OCR baseline visibly fails, with
  the garbled transcript on display.
- **Phase 5: evals + publish.** Full suite green, `make run` works from a
  clean clone, publish to `qdrant-labs`.

## Eval plan (runnable scripts in `evals/`)

1. **Golden retrieval set:** 25 queries authored after Phase 2 by sampling
   real pages (at least 8 targeting handwriting or stamps): target page in
   top 5 for ≥ 80%.
2. **OCR shootout:** same 25 queries against the baseline; report both hit
   rates as a table in the README. Expect the baseline to win on clean
   typewritten text sometimes: report that honestly; the story is the
   handwriting subset.
3. **Quantization fidelity:** top-5 overlap between binary-quantized+rescored
   and a full-precision control run ≥ 90% across the golden set.
4. **Latency:** p95 < 2.5 s end to end on the demo laptop (encode + prefetch +
   rescore); report the split.
5. **Cold boot:** `make run` from a clean clone (models cached) to a working
   query in < 5 minutes.

## What NOT to build

- No OCR in the primary retrieval path. The baseline exists to lose on
  handwriting, not to backstop the demo.
- No PDF-parsing framework, no LangChain/LlamaIndex, no chunking libraries.
  Pages are images.
- No chat interface, no conversation history. One query box, answers with
  evidence.
- No hosted encoder, no Modal/Replicate integration in v1: document it as
  future work in the README and stop there.
- No fine-tuning, no training, no model comparison matrix.

## Demo script (ship this in the README)

1. Query for typed content ("the memo about the marketing budget"): instant
   results, heatmap on the matching paragraph.
2. Point at the memory meter: 4.9 GB of patch vectors held as ~150 MB, same
   answers, rescoring on disk originals.
3. Hero: query targeting a handwritten annotation. Heatmap lights the ink.
   Toggle split view: OCR baseline returns garbage, show its garbled
   transcript.
4. Close: ~12.7k pages, ~750 patch vectors each, one engine doing MaxSim,
   quantization, and tiered storage. No OCR in the retrieval path.

## Publish

- Repo: `qdrant-labs/<pick-a-name>` (public, Apache-2.0). README with demo
  script, architecture diagram, `make setup` / `make run` / `make reset`,
  dataset licensing notes (UCSF Industry Documents Library terms), eval
  results table, and the future hosted-encoder + Vercel path.

## Copy rules for everything public (README, UI, captions)

- Qdrant is a **vector search engine**. Use that exact category.
- Problem first, numbers over adjectives; every number on screen must be
  measured by this repo. Never use: seamless, powerful, robust, cutting-edge.
- No em dashes in public copy. Active voice. Title Case for headings and
  buttons. American English, Oxford comma. Contractions are fine.
- The corpus is a public records archive: never present it as real insurance
  or medical data.
