from __future__ import annotations


class ProgressBar:
    """Minimal single-line console progress bar for downloads and enrichment."""

    def __init__(self, total: int, *, label: str, width: int = 28) -> None:
        self.total = max(total, 0)
        self.label = label
        self.width = width
        self.last_text = ""

    def update(self, current: int) -> None:
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
        if self.last_text:
            print()
