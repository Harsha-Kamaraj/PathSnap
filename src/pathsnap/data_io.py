"""Loading the raw PathSnap trip files.

Raw layout (verified in EDA, see outputs/eda/EDA_REPORT.md):

    <DATA_ROOT>/<partition>/<trip_id>/<trip_id>.track   lon lat t          (float, no header)
                                     <trip_id>.nodes   lon lat            (float, row index = node_id)
                                     <trip_id>.arcs    from_node to_node  (int,   row index = arc_id)
                                     <trip_id>.route   arc_id             (int,   chronological)

Every trip ships its OWN road-network extract, so node_id / arc_id are only
meaningful inside that trip.

By default the data is read from the copy committed in the repo
(`<repo>/Student folder/{train,student_test}`). To use another location, set
PATHSNAP_DATA to a folder that contains the partition sub-folders.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "Student folder"
FILE_TYPES = ("track", "nodes", "arcs", "route")
# expected number of whitespace-separated columns per file type
EXPECTED_COLS = {"track": 3, "nodes": 2, "arcs": 2, "route": 1}


def data_root() -> Path:
    root = Path(os.environ.get("PATHSNAP_DATA", DEFAULT_DATA_ROOT))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: {root}. Set PATHSNAP_DATA to the folder holding train/ and student_test/."
        )
    return root


def list_partitions(root: Path | None = None) -> list[str]:
    root = root or data_root()
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_trips(partition: str, root: Path | None = None) -> list[str]:
    root = root or data_root()
    return sorted(p.name for p in (root / partition).iterdir() if p.is_dir())


@dataclass
class ParseReport:
    """What we found while parsing one raw file (used for data-quality checks)."""
    exists: bool = False
    size_bytes: int = 0
    n_rows: int = 0
    n_blank_rows: int = 0
    n_bad_colcount: int = 0      # rows whose column count differs from EXPECTED_COLS
    n_non_numeric: int = 0       # rows that failed numeric conversion
    n_non_finite: int = 0        # rows containing nan / inf
    col_counts: list = field(default_factory=list)  # distinct column counts seen


def parse_file(path: Path, kind: str) -> tuple[np.ndarray, ParseReport]:
    """Robust line-by-line parse that records problems instead of crashing."""
    rep = ParseReport()
    ncol = EXPECTED_COLS[kind]
    if not path.exists():
        return np.empty((0, ncol)), rep
    rep.exists = True
    rep.size_bytes = path.stat().st_size
    rows, counts = [], set()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                rep.n_blank_rows += 1
                continue
            rep.n_rows += 1
            counts.add(len(tok))
            if len(tok) != ncol:
                rep.n_bad_colcount += 1
                continue
            try:
                vals = [float(t) for t in tok]
            except ValueError:
                rep.n_non_numeric += 1
                continue
            if not np.all(np.isfinite(vals)):
                rep.n_non_finite += 1
            rows.append(vals)
    rep.col_counts = sorted(counts)
    arr = np.asarray(rows, dtype=float).reshape(-1, ncol)
    return arr, rep


@dataclass
class Trip:
    partition: str
    trip_id: str
    track: np.ndarray   # (n, 3) lon, lat, t
    nodes: np.ndarray   # (N, 2) lon, lat
    arcs: np.ndarray    # (A, 2) int from, to
    route: np.ndarray   # (R,)  int arc ids (may be empty if not supplied)
    reports: dict       # kind -> ParseReport


def load_trip(partition: str, trip_id: str, root: Path | None = None) -> Trip:
    root = root or data_root()
    d = root / partition / trip_id
    arrays, reports = {}, {}
    for kind in FILE_TYPES:
        arrays[kind], reports[kind] = parse_file(d / f"{trip_id}.{kind}", kind)
    return Trip(
        partition=partition,
        trip_id=trip_id,
        track=arrays["track"],
        nodes=arrays["nodes"],
        arcs=arrays["arcs"].astype(np.int64),
        route=arrays["route"].reshape(-1).astype(np.int64),
        reports=reports,
    )
