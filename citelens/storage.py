"""Where PDF bytes live: a Storage Protocol, with LocalStorage now and S3Storage later.

The rest of citelens only says ``storage.put(key, data)`` / ``storage.get(key)`` and
never learns whether the bytes sit in a folder, in memory, or in an S3 bucket.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol

# A key is the SHA-256 of the PDF's bytes, as 64 lowercase hex characters.
_KEY = re.compile(r"^[0-9a-f]{64}$")


def pdf_key(data: bytes) -> str:
    """The key a PDF is stored under: identical bytes always give the identical key."""
    return hashlib.sha256(data).hexdigest()


class Storage(Protocol):
    """Anything that can store PDF bytes under a key and give them back."""

    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes:
        """Return the bytes stored under ``key``; raise KeyError if there are none."""
        ...


class LocalStorage:
    """Storage in a folder on this computer: one ``<key>.pdf`` file per PDF.

    Doesn't inherit from Storage; mypy accepts it because its methods match.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> None:
        self._path(key).write_bytes(data)

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise KeyError(key)
        return path.read_bytes()

    def _path(self, key: str) -> Path:
        # Keys will come from outside (uploads, API calls). Only accepting hex
        # means a key like "../../secrets" can't reach outside the folder.
        if not _KEY.match(key):
            raise ValueError(f"not a storage key: {key!r}")
        return self._root / f"{key}.pdf"
