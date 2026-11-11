from typing import NamedTuple


class Chapter(NamedTuple):
    """One chapter. `text` holds one paragraph per line."""
    title: str
    text: str
