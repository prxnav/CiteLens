from pathlib import Path

import pytest

from citelens.storage import LocalStorage, Storage, pdf_key

PDF = b"%PDF-1.7 pretend this is a paper"


class MemoryStorage:
    """A test fake: a dict. Like LocalStorage, it never mentions Storage."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._files[key] = data

    def get(self, key: str) -> bytes:
        return self._files[key]


def store_and_fetch(storage: Storage, data: bytes) -> bytes:
    """Stands in for citelens code: it only knows it was given *a* Storage."""
    key = pdf_key(data)
    storage.put(key, data)
    return storage.get(key)


def test_same_bytes_same_key_different_bytes_different_key() -> None:
    assert pdf_key(PDF) == pdf_key(bytes(PDF))
    assert pdf_key(PDF) != pdf_key(PDF + b" v2")
    assert len(pdf_key(PDF)) == 64


def test_local_storage_round_trip(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    key = pdf_key(PDF)
    storage.put(key, PDF)
    assert storage.get(key) == PDF
    assert (tmp_path / f"{key}.pdf").read_bytes() == PDF


def test_storing_twice_keeps_one_file(tmp_path: Path) -> None:
    storage = LocalStorage(tmp_path)
    storage.put(pdf_key(PDF), PDF)
    storage.put(pdf_key(PDF), PDF)
    assert len(list(tmp_path.iterdir())) == 1


def test_missing_key_raises_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        LocalStorage(tmp_path).get(pdf_key(b"never stored"))


@pytest.mark.parametrize("key", ["../escape", "paper.pdf", "A" * 64, ""])
def test_rejects_keys_that_are_not_hashes(tmp_path: Path, key: str) -> None:
    storage = LocalStorage(tmp_path / "store")
    with pytest.raises(ValueError):
        storage.put(key, PDF)
    with pytest.raises(ValueError):
        storage.get(key)
    assert not (tmp_path / "escape").exists()


def test_any_storage_works_where_storage_is_expected(tmp_path: Path) -> None:
    # mypy checks both calls: each class must match Storage's methods exactly.
    assert store_and_fetch(LocalStorage(tmp_path), PDF) == PDF
    assert store_and_fetch(MemoryStorage(), PDF) == PDF
