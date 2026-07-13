import ast
import inspect
from pathlib import Path

from streamlit_recommenders.data import media
from streamlit_recommenders.data.media import item_placeholder


def test_item_placeholder_is_bundled_file():
    placeholder = item_placeholder()
    assert not placeholder.startswith("http")
    assert Path(placeholder).is_file()


def test_data_media_does_not_import_layouts():
    # Guards against reintroducing the data -> layouts -> data import cycle.
    tree = ast.parse(inspect.getsource(media))
    imported = [
        name
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    ]
    assert not any("layouts" in name for name in imported)
