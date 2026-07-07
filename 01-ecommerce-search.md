# Demo 1: Best-in-Class E-Commerce Search

**Build brief. Self-contained: everything you need is in this file. Read it fully before writing any code.**

## What you are building

A cinematic fashion storefront over 2M+ real products where every shelf is **one
Qdrant request**: hybrid search (dense + miniCOIL sparse), one-stage faceted
filters, and a formula query that fuses relevance, popularity, freshness, margin,
stock, and the shopper's taste vector server-side. Product pages pull "similar"
and "visually similar" shelves from the same engine via the Recommend API. A
merchandiser desk overlays it all: sliders and saved campaign presets, with the
scoring formula always on screen as live code.

**The hero moment (this must work flawlessly):** a merchandiser flips on the
*Winter High-Margin Push* campaign for a returning shopper. The formula-query
expression updates on screen as live code, and the entire shelf reorders in a
single server-side request: relevance, margin, freshness, stock, and that
shopper's taste fused by one engine.

**Why this demo exists:** on Pinecone, Weaviate, Milvus, Redis, or pgvector this
takes an app-side reranker for signal fusion, a separate recommender, or a
RAM-priced cluster at this scale. In Qdrant the fusion, hybrid retrieval,
filtering, and recommendations run in one engine, in one request. The demo's job
is to make that visible.

**Acceptance bar:** deployed on Vercel, repo published to the `qdrant-labs`
GitHub org, all evals in this file passing, the demo script reproducible
end to end.

## How to work (instructions for the building agent)

1. Work through the phases in order. Each phase ends with a **gate**: run it,
   make it pass, only then continue.
2. **Verify request syntax against the linked Qdrant docs before coding each
   query.** This brief pins the architecture; exact field names in the Query API
   evolve. Verified as of 2026-07: formula queries (v1.14+, `$score`,
   `exp_decay`, `mult`, `sum`), weighted RRF (v1.17+).
   Docs: https://qdrant.tech/documentation/concepts/hybrid-queries/
3. Use executive judgment. This brief exists to produce a great demo that
   showcases Qdrant, not to enforce every literal implementation detail. If a
   detail blocks stage impact, make the smallest sensible decision, document it
   in the README, and preserve the Qdrant story.
4. Keep going until the project is complete. Build through all phases, run the
   gates and evals, deploy or publish as far as access allows, and do not stop at
   a prototype unless an external access blocker makes completion impossible.
5. Keep it lean. One Next.js app, Python scripts for ingest, Qdrant as the only
   backend. The "What NOT to build" section is binding.
6. If an access item under Prerequisites is missing, document the blocker in the
   README, make any progress that does not depend on it, and do not invent a
   weaker substitute that breaks the demo story.

## House rules

- Qdrant is a **vector search engine**. Use that exact category in any UI copy,
  README, code comment, or commit in this project.
- This scope is a brief, not a contract. Make executive decisions when a detail
  gets in the way of a better demo, document the decision, and keep the Qdrant
  story intact.
- The acceptance bar is completion. Work through all phases, run the gates and
  evals, and deploy or publish as far as access allows.
- Respect the "What NOT to build" section. The demo is judged on stage impact and
  legibility, not feature count.

## Prerequisites

- Qdrant Cloud account (company org), permission to create a **4 GB / 2 vCPU
  starting cluster** and scale it up if the full-ingest gate misses RAM or
  latency. The target story is "millions on a modest node," not "stay on the
  smallest node at all costs." Record the final cluster size in the README.
  Cloud Inference is enabled by default on new clusters.
- GitHub write access to `qdrant-labs`; Vercel team membership.
- Hugging Face account (free) for the dataset download.
- A machine that can run a multi-hour Python ingest (laptop is fine).

## Locked architecture (do not relitigate these)

| Decision | Choice | Why |
|---|---|---|
| Catalog | Amazon Reviews 2023, **Clothing, Shoes & Jewelry** item metadata, 2M+ items | Image-rich, seasonal merchandising ("winter stock") reads naturally |
| Dense vectors | 384-d text model available in **both** FastEmbed and Cloud Inference (prefer `sentence-transformers/all-MiniLM-L6-v2`; confirm in the cluster's Inference tab) | Bulk ingest embeds locally (free, fast); **query-time text embeds in-cluster via Cloud Inference** so the serving path shows Qdrant doing the embedding |
| Sparse vectors | miniCOIL (`Qdrant/minicoil-v1` in FastEmbed) | Hybrid branch; term-aware sparse scoring |
| Image vectors | CLIP, **200k flagship subset only** (the items the demo script visits + top-rated per category) | "Visually similar" shelf wows; embedding all 2M images costs days and ~100 GB egress for no extra stage value |
| Quantization | Scalar int8, originals on disk | The 4 GB node is the first target; scale up only if measured RAM, payload indexes, sparse vectors, or CLIP vectors require it |
| Margin / stock | **Synthesized deterministically** (seeded by item ID) | Not in any public dataset. Seeded → re-ingest reproduces identical values |
| Personas | 6 shopper personas, checked in as `fixtures/personas.json` (purchase history ASINs, returns, taste vector = mean of purchased items' dense vectors) | Demo boots from fixtures, identical every run |
| Merchandiser formulas | Campaign presets + sliders → a small server-side **formula builder** (whitelisted fields and ops, weights clamped) → Qdrant formula JSON, always displayed as live code | The formula is data: inspectable on screen, unit-testable, and every shelf request stays a single Qdrant call |
| Auto-play | Scripted sequence (search → click → preset flip) with pre-written captions | Runs unattended on a booth screen |
| Stack | Next.js (App Router) + shadcn/ui on Vercel; Python + `uv` for ingest scripts | Ingest never runs on Vercel |

## Data

Source: `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face, config
`raw_meta_Clothing_Shoes_and_Jewelry` (item metadata, NOT the reviews). Stream
it; do not download the full dump. Keep items with a non-empty title and at
least one image URL. Target ≥ 2M items.

Payload per point (precompute normalized fields at ingest so formulas stay simple):

```
asin, title, brand (store), category (leaf of categories[]), price,
average_rating, rating_number, image_url (first large image),
margin        - synthesized: seeded uniform by category band, 0.05-0.65
stock         - synthesized: seeded int 0-500, ~8% zero
listed_at     - synthesized datetime, seeded, skewed to last 24 months
pop_norm      - log10(1+rating_number) / log10(1+max), clamped 0-1
margin_norm   - margin rescaled 0-1
```

Handling nulls: `price` is missing for a large share of items. Fill with the
category median and set `price_estimated: true`. Formulas must never touch a
null.

Payload indexes: `category` (keyword), `brand` (keyword), `price` (float),
`average_rating` (float), `margin_norm` (float), `stock` (integer),
`listed_at` (datetime), `pop_norm` (float).

## Qdrant design

One collection, named vectors: `dense` (384-d, int8 scalar quantization,
originals on disk), `clip` (512-d, only on the 200k subset), sparse `minicoil`.

**Shelf query (the core request), one ranked-shelf round trip:**

1. Prefetch branch A: `dense` against the query text (query-time embedding via
   Cloud Inference `Document` object).
2. Prefetch branch B: `minicoil` against the query text.
3. Prefetch branch C: `dense` against the active persona's taste vector.
4. Fuse with **weighted RRF** (start `[1.0, 1.0, 0.6]`; retune only against the
   golden-query eval set).
5. Top-level **formula query** over the fused score. Implement this as a nested
   Query API request: inner prefetches produce the RRF candidate set, and the
   outer query rescoring formula runs over that set.

```json
{
  "prefetch": {
    "prefetch": [
      { "query": { "document": { "text": "<query>", "model": "<dense-model>" } }, "using": "dense", "limit": 200 },
      { "query": { "document": { "text": "<query>", "model": "<sparse-model>" } }, "using": "minicoil", "limit": 200 },
      { "query": "<persona_taste_vector>", "using": "dense", "limit": 200 }
    ],
    "query": { "rrf": { "weights": [1.0, 1.0, 0.6] } },
    "limit": 100
  },
  "query": {
    "formula": {
      "sum": [
        "$score",
        { "mult": [0.35, "margin_norm"] },
        { "mult": [0.20, "pop_norm"] },
        { "mult": [0.25, { "exp_decay": {
          "x": { "datetime_key": "listed_at" },
          "target": { "datetime": "<now>" },
          "scale": 7776000,
          "midpoint": 0.5
        } } ] },
        { "mult": [-10.0, { "key": "stock", "range": { "lte": 0 } }] }
      ]
    },
    "defaults": { "margin_norm": 0.0, "pop_norm": 0.0 }
  },
  "filter": "<facets>",
  "limit": 24
}
```

The `w_*` values come from the merchandiser sliders or the active campaign
preset (substitute real numbers server-side; the stock-gate condition syntax
must be verified against the score-boosting docs). Facet filters (category,
brand, price range, rating) go in the request's `filter` and are applied during
HNSW traversal, not post-hoc. The ranked shelf is one Query API request. Facet
counts may use separate bounded Qdrant facet requests; keep those requests
visible in the latency panel so "one request" never becomes a hidden claim.

**Product-page shelves via the Recommend API:**
- "Similar": recommend, positives = this item, `using: dense`.
- "Visually similar": recommend, positives = this item, `using: clip`
  (subset items only; hide the shelf otherwise).
- "Shoppers like you also bought": recommend with `strategy: best_score`,
  positives = persona purchase history, negatives = persona returns.

**The formula builder:** presets and sliders map to a small AST: fields
`{margin_norm, pop_norm, listed_at, price, average_rating, stock}`, ops
`{sum, mult, exp_decay, condition}`, weights clamped to [-10, 10]. The
server validates the AST (whitelist walk), translates it to Qdrant formula
JSON, and the UI shows the result as live code with a before/after diff. Ship
five campaign presets as fixtures ("Winter High-Margin Push", "New Arrivals
First", "Clearance Mode", ...), each a named, human-readable AST. Include 10+
unit-test fixtures covering valid presets, weight clamping, and rejection of
unknown fields.

## UI (this is a showcase, not an admin panel)

- **Storefront:** full-bleed product wall, instant search-as-you-type, facet
  sidebar with live counts, a latency chip on every shelf ("38 ms · 1 Qdrant
  request"), smooth FLIP reorder animation when the formula changes.
- **Merchandiser desk:** slide-over panel. Sliders (relevance / margin /
  popularity / freshness), the campaign preset picker, and the **formula always
  visible as code** with a before/after diff flash on every change.
- **Product page:** hero image, payload facts, the three Recommend shelves.
- **Persona switcher:** header dropdown; switching personas visibly reorders
  the same query (this is the taste-vector proof).
- **Auto-play:** toggle in the header; runs the scripted loop with pre-written
  captions; ESC exits instantly.
- Dark, editorial visual design. This page will be projected on stage:
  typography and motion matter as much as latency.

## Build phases and gates

- **Phase 0: access check.** Create the cluster; confirm in the console
  Inference tab which dense + sparse models exist; record choices in `.env`.
  Gate: a scripted upsert + query with a Cloud Inference `Document` succeeds.
- **Phase 1: schema + smoke ingest.** Collection, indexes, ingest script with
  `--limit`. Gate: 5k items in, hybrid shelf query returns visually sane
  results.
- **Phase 2: full ingest.** 2M+ items, batched (256), checkpointed and
  resumable, embeddings via local FastEmbed. Gate: point count ≥ 2M, p95 shelf
  query < 150 ms server-side, cluster RAM below 80%. If the gate fails because
  the starting cluster is too small, scale the cluster once, rerun the gate, and
  record both measurements in the README.
- **Phase 3: CLIP subset.** Select 200k flagship items, download images with
  retry + dead-link skip, embed, update points. Gate: visually-similar shelf
  returns same-garment-style items for 10 spot checks.
- **Phase 4: storefront UI.** Gate: golden queries (eval set) look right;
  latency chip live.
- **Phase 5: merchandiser desk + formula builder.** Gate: builder unit tests
  pass; the hero preset reorders the shelf end to end.
- **Phase 6: product pages + Recommend shelves.** Gate: recommend smoke tests.
- **Phase 7: auto-play.** Gate: 5-minute unattended run, no errors, fallback
  mode works with the network throttled.
- **Phase 8: evals, deploy, publish.** Gate: full eval suite green on the
  Vercel deployment, then publish to `qdrant-labs`.

## Eval plan (write these as runnable scripts in `evals/`)

1. **Golden queries** (~30, checked in as JSON): query → ASINs expected in
   top 10. Pass ≥ 80%. Author them after Phase 2 by browsing real data.
2. **Persona differentiation:** same query, two personas → Spearman rank
   correlation of top 50 < 0.7. Proves the taste branch does something.
3. **Formula builder suite:** preset fixtures → expected formula JSON;
   out-of-range weights clamped; unknown fields rejected.
4. **Latency budget:** p95 < 150 ms server-side per shelf at full scale,
   measured from the Vercel region.
5. **Facet correctness:** counts under an active filter match a scroll-based
   recount on three spot checks.
6. **Stage rehearsal checklist:** scripted walkthrough of the demo script
   below; every step must succeed twice in a row.

## What NOT to build

- No auth, no user accounts, no cart/checkout, no CMS.
- No database other than Qdrant. Personas and golden sets are JSON fixtures.
- No reranker service, no embedding microservice, no queue. Ingest is a script.
- No image vectors beyond the 200k subset (document full-catalog CLIP as a
  scale-up path in the README instead).
- Don't chase exotic UI state management; React state + URL params suffice.

## Demo script (ship this in the README)

1. Type "waterproof hiking boots": hybrid results, latency chip, facet counts.
2. Filter brand + price: one-stage filtering, counts update.
3. Open the merchandiser desk, drag margin up: shelf reorders, formula diff.
4. Hero: flip on the "Winter High-Margin Push" campaign. The formula updates
   as live code and the shelf reorders in one request.
5. Switch persona: same query, different shelf.
6. Open a product: three Recommend shelves, point at "visually similar".
7. Toggle auto-play and walk away.

## Publish

- Repo: `qdrant-labs/<pick-a-name>` (public, Apache-2.0). Include README with
  demo script, architecture diagram, one-command setup (`uv run ingest`,
  `vercel deploy`), and eval instructions.
- Vercel: production deployment, env vars documented.

## Copy rules for everything public (README, UI, captions)

- Qdrant is a **vector search engine**. Use that exact category.
- Problem first, numbers over adjectives ("one request, 38 ms" beats vague
  speed claims). Never use: seamless, powerful, robust, cutting-edge.
- No em dashes in public copy. Active voice. Title Case for headings and
  buttons. American English, Oxford comma. Contractions are fine.
- Don't claim benchmark numbers you didn't measure in this repo.
