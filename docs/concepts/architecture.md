# Architecture

![System overview](../figures/architecture.svg)

## The session loop

You provide the tables and a recommender. The library keeps the session profile, calls the
recommender through a cache, and renders the result. Clicks, likes, dislikes and skips go back
into the profile, and the next refresh scores against it.

```mermaid
flowchart LR
  subgraph YOU["you provide"]
    d["items · interactions"]
    f["get_recommendations()"]
  end
  subgraph LIB["the library"]
    s["session profile"]
    c["cache + adapter"]
    lay["layouts: rows · grid · cards"]
  end
  YOU --> LIB --> UI["browser"]
  UI -. "click · like · dislike · skip" .-> s
```

One refresh: sidebar params and the selected user go to the cached `get_recommendations()`
(through the adapter, which drops arguments the function does not declare); the returned ids are
stored per section in session state and rendered. **Get Recommendations** recomputes every section
against the same profile, which is what makes compared models comparable.

## Module map

```mermaid
flowchart TB
  run["runner.py · sr.run"]
  data["data/ · loaders, prepare (MovieLens, goodbooks, TMDB)"]
  subgraph contract["contract"]
    models["models/ · protocol, adapter"]
    recs["recommenders/ · BaseRecommender, ArtifactRecommender"]
  end
  subgraph runtime["runtime/ (internal)"]
    rt["cache · state · seen · compare · keys"]
  end
  subgraph present["presentation"]
    layouts["layouts/ · rows, grid, cards"]
    widgets["widgets/ · params, profile strips"]
    viz["viz/ · plots, dataset_info"]
    extras["metrics/ · evaluate   content/ · markdown"]
  end
  config["config/ · YAML overrides"] --> run
  data --> run
  run --> contract
  run --> runtime
  run --> present
```

Training is deliberately outside: the library standardizes how a trained recommender is shown and
probed, not how it is trained.
