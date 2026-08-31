from __future__ import annotations

import argparse
from pathlib import Path

from investrag.config import Settings
from investrag.service import InvestRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local files into InvestRAG Studio")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    service = InvestRAGService(Settings())
    for path in args.paths:
        source = service.ingest(path)
        print(source.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
