import pandas as pd

from streamlit_recommenders.layouts._helpers import visible_entries


def test_visible_entries_deduplicates_rec_ids():
    items = pd.DataFrame(
        {
            "item_id": [1, 2, 18],
            "title": ["A", "B", "C"],
            "image_url": ["", "", ""],
            "description": ["", "", ""],
        }
    )
    entries = visible_entries(items, [18, 1, 18, 2], selected_ids=set())
    assert [entry["id"] for entry in entries] == [18, 1, 2]


def test_visible_entries_keeps_selected_items_marked():
    items = pd.DataFrame(
        {
            "item_id": [1, 2, 18],
            "title": ["A", "B", "C"],
            "image_url": ["", "", ""],
            "description": ["", "", ""],
        }
    )
    entries = visible_entries(items, [18, 1, 2], selected_ids={18})

    assert [entry["id"] for entry in entries] == [18, 1, 2]
    assert entries[0]["selected"] is True
    assert entries[1]["selected"] is False
