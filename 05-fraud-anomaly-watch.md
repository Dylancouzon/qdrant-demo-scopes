# Demo 5: Fraud & Anomaly Watch

**Build brief. Self-contained: everything you need is in this file. Read it fully before writing any code.**

## What you are building

A live payments wall for a fictional card network. Synthetic transactions
stream across the screen; fraud ignites red in under a second, and every alert
carries a plain-language explanation grounded in retrieved evidence, not a
black-box score. "Normal" is defined **per customer**: thousands of individual
baselines live in one Qdrant collection via multitenancy, and every new event
is searchable the moment it lands.

**The hero moment (this must work flawlessly):** an audience member opens the
public launch link (or scans a QR code), picks one dramatic "attack card"
(Geo-Hop, Card Testing, or Amount Ladder), and watches the generated transaction
sequence flare red on the wall, with a neighbor-grounded reason, before they
lower the phone.

**Why this demo exists:** per-customer baselines are expensive elsewhere
(per-namespace pricing, table partitioning); in Qdrant, tenant-keyed payload
indexing gives each customer a local data layout inside one collection, and
upserts are searchable immediately with no refresh cycle. The demo's job is to
make both visible.

**Acceptance bar:** deployed on Vercel as a public browser demo, repo published
to `qdrant-labs`, all evals passing, the attack-card hero moment reproducible
from both a desktop browser and a phone that isn't yours.

## How to work (instructions for the building agent)

1. Work through the phases in order; pass each gate before continuing.
2. Verify Qdrant request syntax against the linked docs before coding each
   query. Verified as of 2026-07: formula queries v1.14+ (`$score`,
   `exp_decay`), multitenancy via keyword payload index with `is_tenant: true`.
   Docs: https://qdrant.tech/documentation/guides/multitenancy/ and
   https://qdrant.tech/documentation/concepts/hybrid-queries/
3. Use executive judgment. This brief exists to produce a great demo that
   showcases Qdrant, not to enforce every literal implementation detail. If a
   detail blocks stage impact, make the smallest sensible decision, document it
   in the README, and preserve the Qdrant story.
4. Keep going until the project is complete. Build through all phases, run the
   gates and evals, deploy or publish as far as access allows, and do not stop at
   a prototype unless an external access blocker makes completion impossible.
5. Qdrant is the only backend. No queue, no Redis, no Postgres. If you feel you
   need shared state, re-read the Architecture section: Qdrant IS the shared
   state.
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

- Qdrant Cloud account (company org), a **1–2 GB cluster** is plenty.
- GitHub write access to `qdrant-labs`; Vercel team membership.

## Locked architecture (do not relitigate these)

| Decision | Choice | Why |
|---|---|---|
| Platform | Everything on Vercel; **Qdrant is the only state** | No worker service to babysit; survives redeploys; the finished demo is a public browser experience |
| Stream | An SSE route handler generates events on the fly, **seeded by (world_seed, time_bucket)** with deterministic event IDs | Stateless: a reconnect regenerates the same events for the same bucket, and upserts are idempotent. Determinism is a feature, not a limitation |
| Wall updates | The wall's SSE loop both emits generator events and, each tick, picks up recent browser-launched attack events from Qdrant (timestamp-filtered scroll) | Audience launches may hit a different serverless instance than the wall's stream; Qdrant is the bus |
| Features | **Engineered ~31-d vector, no learned encoder** (spec below) | Deterministic and human-readable, which is what makes the evidence panel honest |
| Distance | Euclidean | Feature deltas are meaningful in absolute terms |
| Anomaly score | Self-normalizing kNN ratio (LOF-style, spec below), k = 10, computed **within the event's tenant** | Stateless, per-tenant by construction: $2k is routine for one card and an alarm for another |
| Cold start | First **30 transactions per tenant** are a learning window: scored but never alerted | Without this every new tenant screams red |
| Browser launch flow | A responsive `/launch` page assigns the visitor a **pre-seeded persona** ("you are Customer #4711; here is their normal life") and shows three big attack cards: Geo-Hop, Card Testing Burst, Amount Ladder | A fresh tenant has no baseline, so every attack card targets a pre-seeded customer. One tap has more stage energy than a form, and a shareable URL works better online than a QR-only path |
| Explanations | Deterministic, computed from the actual neighbor stats: a one-liner on the wall ("first card-present event outside the EU in 240 transactions; nearest normal neighbor is 6× closer to baseline") built from a small template library keyed on which feature blocks deviate most, plus the full math in the evidence panel | Grounded templates over real retrieved evidence read credibly on stage and work offline |
| Recency | Formula query with `exp_decay` on the timestamp when retrieving neighbors, with Euclidean scores handled explicitly | Stale behavior fades from "normal" without inverting closest-neighbor order |
| Stack | Next.js (App Router) + shadcn/ui; a small `scripts/seed.ts` for baselines | One repo, one deploy, one public URL |

## The synthetic world

- **200 tenants** (customers), each with a seeded behavior profile: home geo,
  2–4 frequented merchant categories, log-normal amount distribution, active
  hours, online/card-present mix.
- `scripts/seed.ts` generates and upserts **300–800 baseline transactions per
  tenant** (~100k points) before the demo ever runs.
- The live generator emits ~5 events/second across random tenants, following
  their profiles.
- **Three injected fraud motifs**, each with a ground-truth label in the
  payload (`motif: card_testing | geo_hop | ladder | none`; the scorer must
  never read this field, evals do):
  - *Card testing:* burst of small same-merchant online charges in seconds.
  - *Geo-hop:* card-present events in two cities with impossible travel time.
  - *Amount laddering:* escalating amounts at the same merchant, minutes apart.

The motif scorer needs a small amount of recent-history context. Compute these
deterministically before vectorization by scrolling the tenant's recent events
from Qdrant:

```
minutes_since_last_tx        clipped to 0-1440, log-scaled
same_merchant_10m_count      clipped to 0-10
impossible_travel_kmh        clipped to 0-2000, then /2000
amount_ratio_recent_median   clipped to 0-10, log-scaled
ladder_step_ratio            current amount / previous same-merchant amount
```

These are retrieval features, not labels. The scorer still never reads `motif`.

## Feature vector (~31 dims, weights are part of the spec)

```
log10(1+amount)/5           ×2.0   (1 dim)
sin/cos hour-of-day         ×1.0   (2)
sin/cos day-of-week         ×0.5   (2)
geo → unit-sphere xyz       ×1.5   (3)
merchant category one-hot   ×1.0   (12)
channel one-hot pos/online/atm ×1.0 (3)
card_present flag           ×1.0   (1)
currency-is-home flag       ×1.0   (1)
new-merchant-for-tenant flag ×1.5  (1)
recent-history features      ×1.0  (5)
```

Weights balance the blocks' scales; tune only via the eval suite, and record
any change in the README.

**Anomaly score (stateless, self-normalizing):** retrieve the event's k = 10
nearest neighbors filtered to its tenant. Use a prefetch nearest-neighbor query,
then rescore with a formula that adds recency. Because Euclidean distance sorts
closest points by lower distance while formula queries sort larger scores first,
normalize or negate the distance component explicitly before mixing it with
`exp_decay`; verify the final request against the current Search Relevance docs.
Let `d_event` = mean distance from the event to those neighbors, and `d_local` =
mean distance from those neighbors to their own centroid. Score = `d_event /
d_local`. Alert when score > threshold (start at 2.5, tune via evals). This is a
simplified local-outlier-factor: it needs no per-tenant statistics stored
anywhere, so it works on stateless serverless instances.

Canonical neighbor query shape:

```json
{
  "prefetch": {
    "query": "<event_vector>",
    "using": "features",
    "filter": {
      "must": [
        { "key": "tenant_id", "match": { "value": "<tenant_id>" } }
      ]
    },
    "limit": 100
  },
  "query": {
    "formula": {
      "sum": [
        { "mult": [-1.0, "$score"] },
        { "mult": [0.15, { "exp_decay": {
          "x": { "datetime_key": "ts" },
          "target": { "datetime": "<event_ts>" },
          "scale": 2592000,
          "midpoint": 0.5
        } } ] }
      ]
    }
  },
  "with_vector": true,
  "limit": 10
}
```

The response score is now the recency-adjusted ranking score, not the raw
Euclidean distance. Recompute `d_event` client-side from the returned neighbor
vectors so the evidence panel can show the exact distance arithmetic.

## Qdrant design

One collection, named vector `features`: 31-d vectors, Euclidean. Payload:
`tenant_id` (keyword index with `is_tenant: true`), `ts` (datetime index),
`amount`, `merchant`,
`merchant_cat`, `city`, `channel`, `motif` (ground truth, evals only),
`channel_src: generator | browser_attack`, `score`, and the five recent-history feature
values (written back so the wall and evidence panel can replay).

Every event: vectorize → query k = 10 with `filter: tenant_id` (+ recency
formula) → score → upsert with score in payload → emit to the wall. Attack cards
may generate one event (Geo-Hop) or a short burst of 3-6 events (Card Testing
Burst, Amount Ladder). Note on stage: points are searchable immediately; small
unindexed segments are scanned exactly as advertised, no refresh cycle exists.

## UI

- **The wall:** dark, full-screen animated stream: events pulse through as dots
  grouped by tenant; anomalies ignite red with the deterministic one-liner
  ("first card-present event outside the EU in 240 transactions; nearest
  normal neighbor is 6× closer to baseline"). A ticker shows events/sec,
  p95 score latency, and total points, live.
- **Evidence panel** (click an alert): the tenant's baseline as a 2-D scatter
  (project the tenant's points with PCA client-side; ~800 points is trivial),
  the neighbor list with actual distances, and the score arithmetic laid out
  step by step.
- **Public routes:** `/` is the wall, `/launch` is the browser-friendly attack
  launcher, `/alert/[id]` deep-links to an evidence panel, and `/join` redirects
  to `/launch` for old QR links.
- **Attack launcher** (`/launch`): works on desktop and mobile. It assigns a
  pre-seeded persona, shows "their normal life" in three lines, then presents
  three attack cards:
  **Geo-Hop**, **Card Testing Burst**, and **Amount Ladder**. One tap launches
  the chosen deterministic sequence. The response shows the attack's live status,
  the highest score, and a short "watch the wall" cue.
- **Online mode:** if no projected wall is open, the launch page opens a compact
  embedded wall strip after launch so the public demo still makes sense from a
  single browser tab.
- The wall auto-reconnects its SSE stream (Vercel function duration ends it
  periodically); reconnection must be visually seamless.

## Build phases and gates

- **Phase 0: access + skeleton.** Cluster up, collection created, one scripted
  upsert + tenant-filtered query works.
- **Phase 1: world + seed.** Generator library + `seed.ts`. Gate: 100k points,
  a tenant's baseline scatter looks like clustered daily life, not noise.
- **Phase 2: scoring path.** Vectorize, retrieve recent tenant context, kNN,
  score, upsert. Gate: scoring eval (below) passes on a seeded run; p95
  event→score < 300 ms.
- **Phase 3: the wall.** SSE loop + animation + one-liners. Gate: 10-minute
  unattended run with reconnects, no duplicates (idempotent IDs), no drift.
- **Phase 4: evidence panel.** Gate: click any alert, the math shown
  reproduces the score exactly and the one-liner matches the evidence.
- **Phase 5: browser attack cards.** Gate: hero moment on a desktop browser and
  a real phone: open `/launch`, choose an attack card, launch sequence, flare
  < 1 s after the first scored event. QR is a shortcut, not the only path.
- **Phase 6: evals, deploy, publish.** Full suite green on the Vercel
  deployment, then publish to `qdrant-labs`.

## Eval plan (runnable scripts in `evals/`)

1. **Motif detection:** seeded run of 5k events containing 60 labeled frauds
   (20 per motif). Recall ≥ 0.8, precision ≥ 0.6 at the shipped threshold.
   Report the confusion table in the README honestly; do not claim a
   fraud-detection rate this demo hasn't measured.
2. **Cold start:** zero alerts fired inside any tenant's 30-event learning
   window across the seeded run.
3. **Determinism:** two runs with the same world seed produce identical alert
   sets.
4. **Latency:** p95 launch→first-score→flare < 1 s measured end to end from a
   desktop browser and a phone on conference-grade wifi (throttle to simulate).
5. **Tenant isolation:** an event scored against the wrong tenant's baseline
   (deliberate test) produces a different score: proves the filter is doing
   the work.
6. **Public browser smoke:** production Vercel URL works from a clean browser
   profile with no local services, no camera permission, and no preloaded state.

## What NOT to build

- No Kafka, no Redis, no queues, no cron. The SSE route + Qdrant is the whole
  backend.
- No ML training, no learned embeddings, no feature store. The encoder is a
  pure function.
- No auth, no persistence of audience identities beyond the persona
  assignment.
- No free-form transaction composer. The browser flow should feel like launching
  a scenario, not filling out a payment form.
- No fraud-ops dashboard, case management, or historical analytics. It's a
  wall, an evidence panel, and a browser launcher.
- Don't add more motifs, tenants, or features until every eval passes.

## Demo script (ship this in the README)

1. Open the wall: events flowing, ticker live.
2. Point at a quiet tenant's rhythm, then a red flare: read the one-liner.
3. Click the alert: baseline scatter, neighbor distances, the division that
   produced the score.
4. Hero: share the launch link or show the QR. Persona assigned, attack card
   launched, flare before the phone drops.
5. Mention: every event was searchable the instant it landed, one collection,
   200 tenants, one baseline each.

## Publish

- Repo: `qdrant-labs/<pick-a-name>` (public, Apache-2.0). README with demo
  script, architecture diagram, seed + deploy instructions, eval results
  table.
- Vercel: production deployment, env vars documented, public URL included in the
  README. The demo must work from a normal browser tab without installing
  anything.

## Copy rules for everything public (README, UI, captions)

- Qdrant is a **vector search engine**. Use that exact category.
- Problem first, numbers over adjectives. Never use: seamless, powerful,
  robust, cutting-edge, or vague "AI magic" claims.
- No em dashes in public copy. Active voice. Title Case for headings and
  buttons. American English, Oxford comma. Contractions are fine.
- Never overstate: this is a demo of the retrieval mechanics, not a claim
  about production fraud models. The README says so explicitly.
