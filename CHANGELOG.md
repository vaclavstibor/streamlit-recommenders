# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- `sr.run()` orchestration: one call turns a `get_recommendations` callable, model
  object, or dict of named models (compare mode) into an interactive Streamlit demo.
- Layouts: `rows` (scrollable poster rows), `grid` (catalog grid), and `cards`
  (swipe deck with Like / Dislike / Skip and automatic refresh).
- Built-in reference recommenders: `ItemKNNRecommender`, `EASERecommender`,
  `SequentialCFRecommender`, and `ArtifactRecommender` for exported `.npz` weights,
  all on a shared `BaseRecommender` / `RecommenderProtocol` contract.
- Pandas-first data contract: `Dataset`, `ColumnMap`, loaders, and validation.
- Dataset preparation CLI (`python -m streamlit_recommenders.data.prepare`) for
  MovieLens and goodbooks-10k, with optional TMDB poster enrichment.
- Ranking metrics (`evaluate`, hit rate, recall, NDCG, MRR, coverage) and Plotly
  visualizations (metric comparison, overlap heatmap, score distribution).
- Session-state handling: per-user history, session selections, swipe feedback,
  and parameter widgets (sliders, selectboxes) with YAML config support.

[0.1.0]: https://github.com/vaclavstibor/streamlit-recommenders/releases/tag/v0.1.0
