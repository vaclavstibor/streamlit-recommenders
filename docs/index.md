# streamlit-recommenders

Turn a recommender into an interactive web demo that others can inspect: pick a user, see their
history and the model's recommendations, give feedback, compare models side by side.

You supply pandas tables and a function (or object) that returns ranked item ids. The library
provides the [Streamlit](https://streamlit.io) app around it: session profile, caching, layouts,
model comparison, and helpers for metrics and plots. It contains no models of its own.

![System overview](figures/architecture.svg)

Offline metrics say how a model scores on average; they do not show what it recommends to a given
user or how it reacts to new feedback. A demo shows that, but building one is frontend work. This
library makes it a few lines of Python, so a demo can be published with a paper.

## Contents

| Page | Read it when |
|------|--------------|
| [Getting started](getting-started.md) | you want a running demo |
| [Recommender contract](concepts/recommender.md) | you connect your own model |
| [Data contract](concepts/data.md) | you prepare your own tables |
| [Feedback & session](concepts/feedback.md) | you need to know what Like, Dislike, Skip and seen-filtering do |
| [Inspection views](concepts/inspection.md) | you add metrics or plots below the demo |
| [Architecture](concepts/architecture.md) | you want the internals |
| [Examples](guides/examples.md) | you want runnable code for each plug-in form |
| [Datasets](guides/datasets.md) | you need MovieLens, goodbooks or TMDB posters |
| [Training & artifacts](guides/artifacts.md) | you train elsewhere and load weights |
| [API reference](api.md) | you look up a function |
