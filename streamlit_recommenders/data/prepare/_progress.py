"""Minimal single-line console progress bar for dataset preparation."""

from __future__ import annotations


class ProgressBar:
    """Minimal single-line console progress bar for downloads and enrichment."""

    def __init__(self, total: int, *, label: str, width: int = 28) -> None:
        """Create a progress bar.

        Args:
            total: Total number of units; <= 0 renders a plain counter.
            label: Prefix shown before the bar.
            width: Bar width in characters.
        """
        self.total = max(total, 0)
        self.label = label
        self.width = width
        self.last_text = ""

    def update(self, current: int) -> None:
        """Redraw the bar for the current progress, if the rendered text changed.

        Args:
            current: Units completed so far.
        """
        if self.total <= 0:
            text = f"{self.label}: {current:,}"
        else:
            ratio = min(max(current / self.total, 0.0), 1.0)
            filled = int(self.width * ratio)
            bar = "#" * filled + "-" * (self.width - filled)
            text = f"{self.label}: [{bar}] {current:,}/{self.total:,} ({ratio:.0%})"
        if text != self.last_text:
            print("\r" + text, end="", flush=True)
            self.last_text = text

    def finish(self) -> None:
        """Terminate the progress line with a newline if anything was drawn."""
        if self.last_text:
            print()
