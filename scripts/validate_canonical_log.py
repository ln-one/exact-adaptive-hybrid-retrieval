#!/usr/bin/env python3
"""Validate a canonical ED-WRRF JSONL artifact and its checksum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_runner.validation import validate_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_log(args.log, require_clean=not args.allow_dirty), sort_keys=True))


if __name__ == "__main__":
    main()
