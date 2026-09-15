"""Download the CCFv3 files required by the simST MVP without AllenSDK.

This intentionally avoids AllenSDK because AllenSDK 2.16.x pins numpy<1.24,
which conflicts with the modern scientific Python stack used by simST.

Run
---
    python scripts/download_ccfv3.py --out data/ccfv3

Files
-----
    annotation_10.nrrd
    structure_tree.json

The annotation is the Allen CCFv3 October-2017 10-um volume.  The ontology is
Mouse Brain Atlas StructureGraph id=1.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ANNOTATION_URL = (
    "https://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"
)
STRUCTURE_TREE_URL = "https://api.brain-map.org/api/v2/structure_graph_download/1.json"


def _download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"Already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "simST-CCFv3-MVP/0.3.1"})
    print(f"Downloading {url}")

    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            copied = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)
                if total:
                    pct = 100.0 * copied / total
                    print(
                        f"\r  {copied / 1024**2:8.1f} / {total / 1024**2:8.1f} MiB "
                        f"({pct:5.1f}%)",
                        end="",
                        flush=True,
                    )
            if total:
                print()
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"Saved: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/ccfv3")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _download(ANNOTATION_URL, out / "annotation_10.nrrd", force=args.force)
    _download(STRUCTURE_TREE_URL, out / "structure_tree.json", force=args.force)

    print("\nCCFv3 files are ready:")
    print(f"  annotation:    {out / 'annotation_10.nrrd'}")
    print(f"  structure tree:{out / 'structure_tree.json'}")


if __name__ == "__main__":
    main()
