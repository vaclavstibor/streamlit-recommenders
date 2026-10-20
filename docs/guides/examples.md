# Examples

One runnable example per way of plugging in a model. All of them use the MovieLens sample:

```bash
python -m streamlit_recommenders.data.prepare --dataset ml-latest-small
```

| Plug-in form | Files | When it fits |
|--------------|-------|--------------|
| Plain function | your `app.py` ([Getting started](../getting-started.md#your-first-demo)) | you already have something that ranks items |
| `BaseRecommender` subclass | `reference_recommenders.py` | you write the scoring yourself |
| `ArtifactRecommender` | `train_baseline_artifacts.py`, then `compare_models_rows.py`, `compare_models_grid.py`, `swipe_deck_cards.py` | you train elsewhere and only load weights |

All examples read the dataset folder from `SR_DATA_DIR` (default `data/ml-latest-small`).

## `BaseRecommender`: fit in the app

```bash
streamlit run examples/reference_recommenders.py
```

Fits ItemKNN, EASE and Sequential CF at startup and shows them side by side. Each class only
implements `scores(user_id, session_items, **params)`; copy one as a starting point.

!!! warning "Memory"
    In-memory fitting builds dense float64 item × item matrices: about 0.76 GB each for the
    9,742 movies of the sample, and EASE holds several while solving. On a machine with a few GB
    free this can be killed; use the artifact route below instead.

## `ArtifactRecommender`: train once, then only load

```bash
python examples/train_baseline_artifacts.py --data data/ml-latest-small   # writes artifacts/*.npz
streamlit run examples/compare_models_rows.py    # three models side by side
streamlit run examples/compare_models_grid.py    # one model at a time, chosen in the sidebar
streamlit run examples/swipe_deck_cards.py       # EASE as a swipe deck
```

No training code runs in these apps; they call `sr.load_artifacts({...}, TRAIN)` and nothing else
model-related. The swipe deck adds a *Dislike weight* slider: at 0 a dislike only hides the card,
higher values push down movies similar to it (see [Feedback & session](../concepts/feedback.md)).

## A model trained somewhere else

Models from RecBole, Cornac, LensKit or a paper repository need no artifacts; wrap them in the
contract as shown in
[Training & artifacts](artifacts.md#bring-an-externally-trained-model).
