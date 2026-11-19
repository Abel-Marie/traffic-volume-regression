"""Download and load the raw traffic file, verifying it byte for byte."""

from __future__ import annotations

import gzip
import io
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .artifacts import file_sha256
from .config import Config
from .log import LOGGER


def fetch_bytes(url: str, deadline_seconds: float = 90.0) -> bytes:
    """Download with a deadline for the whole transfer, so a slow connection moves on to the next source
    instead of hanging (a socket timeout alone only catches a connection that stops completely)."""
    start = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": "traffic-volume/1.0"})
    chunks = []
    with urllib.request.urlopen(req, timeout=20) as resp:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            if time.monotonic() - start > deadline_seconds:
                raise TimeoutError(f"download exceeded {deadline_seconds:.0f} seconds")


def csv_bytes_from(payload: bytes, name: str) -> bytes:
    """The original CSV bytes from a .csv, .csv.gz or a .zip that contains either."""
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member = next(m for m in zf.namelist() if m.endswith((".csv", ".csv.gz")))
            return csv_bytes_from(zf.read(member), member)
    if name.endswith(".gz"):
        return gzip.decompress(payload)
    return payload


def download_raw(cfg: Config, dest_dir: Path, repo_root: Path | None = None) -> dict:
    """Try each source in turn (UCI, a pinned public mirror through two hosts, the copy in the repository).

    Every source must produce exactly the same bytes: the SHA-256 checksum of
    the CSV is verified whichever source succeeds.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "traffic.csv"
    attempts = [("url", u) for u in cfg.data.sources]
    if repo_root is not None:
        attempts.append(("repo_mirror", str(Path(repo_root) / cfg.data.mirror_path)))
    errors = []
    for kind, where in attempts:
        try:
            payload = (Path(where).read_bytes() if kind == "repo_mirror"
                       else fetch_bytes(where, cfg.data.download_deadline_seconds))
            target.write_bytes(csv_bytes_from(payload, where))
            digest = file_sha256(target)
            if digest != cfg.data.sha256:
                raise ValueError(f"checksum mismatch: {digest}")
            LOGGER.info("raw data from %s (%s)", kind, where)
            return {"path": str(target), "source_kind": kind, "source": where, "sha256": digest,
                    "sources_that_failed": len(errors)}
        except Exception as exc:  # noqa: BLE001 - try the next source, report all failures
            errors.append(f"{kind} {where}: {exc}")
            LOGGER.warning("data source failed: %s", errors[-1])
    raise RuntimeError("Could not obtain the dataset from any source:\n" + "\n".join(errors))


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV. 'None' in the holiday column is a value meaning no holiday, not a missing value."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df.insert(0, "row_id", range(len(df)))
    return df
