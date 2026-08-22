# Architecture

`streamlit-recommenders` is a thin layer between your recommender and an interactive demo: you
provide recommendations through one small contract, and the library owns everything interactive.
This page zooms in through three levels — the idea, who owns what, and the module map.

![System overview](../figures/architecture.svg)

## Level 1 — The idea

You bring data and a way to produce ranked item ids; you get an app that lets anyone inspect,
compare, and probe the model.

```mermaid
flowchart LR
  data["items · interactions"] --> lib["streamlit-recommenders"]
  model["your recommender<br/>function · subclass · artifact"] --> lib
  lib --> app["interactive app<br/>inspect · compare · probe"]
```

## Level 2 — Who owns what, and the session loop

You own the data and the recommendations. The library owns the session profile, caching,
comparison, metrics, and layouts. What the user does on the page (click, like, dislike, skip) feeds
back into the profile and refreshes the recommendations.

```mermaid
flowchart LR
  subgraph YOU["you provide"]
    d["items · interactions"]
    f["get_recommendations()"]
  end
  subgraph LIB["the library owns"]
    s["session profile & feedback"]
    c["caching + model adapter"]
    cmp["side-by-side comparison"]
    viz["metrics + plots"]
    lay["layouts: rows · grid · cards"]
  end
  subgraph UI["you see & do"]
    cards["poster cards / swipe deck"]
    plots["profile strips · plots · tables"]
  end
  YOU --> LIB --> UI
  UI -. "click · like · dislike · skip" .-> LIB
```

The profile a model scores against is the union of the selected user's **history** and their
**current-session** likes; see **[Feedback & session](feedback.md)** for the exact signals,
seen-filtering, and fallback handling.

## Level 3 — Module map

One `sr.run(...)` call wires four groups together. Nothing model-specific lives in the library:
the contract is just `get_recommendations`, and the reference baselines are in `examples/`.

```mermaid
flowchart TB
  run["runner.py — orchestration<br/>(sr.run)"]
  data["data/ — loaders + prepare<br/>MovieLens · goodbooks · TMDB"]

  subgraph contract["the contract"]
    models["models/ — protocol + adapter"]
    recs["recommenders/ — BaseRecommender · ArtifactRecommender"]
  end
  subgraph internal["runtime/ — internal state"]
    cache["cache"]
    state["state · seen"]
    compare["compare"]
  end
  subgraph present["presentation"]
    layouts["layouts/ — rows · grid · cards"]
    widgets["widgets/ — params · profile"]
    vizmod["viz/ — plots · dataset_info"]
  end

  data --> run
  run --> contract
  run --> internal
  run --> present
```

**Request flow:** sidebar params + selected user → cached `get_recommendations()` (via the adapter)
→ ranked item ids stored in session state → poster carousel / swipe deck. A click updates the
session profile; **Get Recommendations** recomputes every compared row against the same evidence.

## Boundary

The library owns the interactive inspection layer, not model training. Training frameworks
standardize offline experimentation; `streamlit-recommenders` standardizes how a trained
recommender is displayed, compared, and probed with session feedback.

- **Data prep lives in the library** (`streamlit_recommenders/data/prepare/`): `prepare_movielens`
  and `prepare_goodbooks` download and normalize into the local schema, TMDB enrichment is robust
  (retry/backoff, completeness report), and a `dataset.json` manifest (`is_complete`) makes
  preparation idempotent.
- **Models do not.** The library ships only the `BaseRecommender` contract and the
  `ArtifactRecommender` loader. Reference baselines live in `examples/reference_recommenders.py` as
  copy-and-adapt starting points; heavy frameworks stay external, reached through the contract.
