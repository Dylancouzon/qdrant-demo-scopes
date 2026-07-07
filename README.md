# Qdrant Demo Scopes

One-shot build briefs for three showcase demos. Each scope is a self-contained
Markdown file: open a fresh Claude Code session (Fable 5), point it at the file,
and work through it top to bottom. The scopes assume **no** shared context:
every constraint, guardrail, and acceptance test is in the file itself.

| Scope | Demo | Core Qdrant story |
|---|---|---|
| `01-ecommerce-search.md` | Best-in-Class E-Commerce Search | Hybrid search + formula queries + Recommend API on 2M+ products, one engine |
| `05-fraud-anomaly-watch.md` | Fraud & Anomaly Watch | Per-customer baselines via multitenancy + instant-searchable upserts |
| `06-visual-docs-search.md` | Visual Docs Search | ColPali multivectors + MaxSim + binary quantization, no OCR retrieval path |

Demos 1 and 5 must end up **deployed on Vercel**. Demo 5 must work as a public,
browser-friendly demo from a normal URL, with QR as an optional shortcut. Demo 6
runs **locally** (its page-image query encoder needs local GPU/MPS compute; a
hosted-encoder + Vercel path is documented in its scope as future work). All
three repos are published to the **qdrant-labs** GitHub organization. That is
the acceptance bar, not a stretch goal.

## Access you need before starting

Confirm all of these on day one. None are optional for your demo:

- **Qdrant Cloud**: account on the company organization, with permission to
  create and scale a cluster (each scope states its starting size and scale
  gate). Cloud Inference is enabled by default on new clusters; verify the models
  your scope names in the cluster's **Inference** tab before ingesting.
- **GitHub**: write access to the `qdrant-labs` organization (the finished demo
  repo is published there).
- **Vercel**: membership in the team the demo deploys under.
- **Hugging Face account (free)**: the datasets (Amazon Reviews 2023, DocVQA)
  and model weights (ColQwen family, demo 6) are public; a token is only needed
  if you hit anonymous download rate limits or a gated fallback asset.
- **Demo 6 only: an Apple Silicon Mac (or NVIDIA GPU machine)**: the page-image
  encoder runs locally for both ingest and query time.

If any of these are missing, document the blocker, make any progress that does
not depend on it, and do not invent a weaker substitute that breaks the demo
story.
