# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check published bridge asset hashes using only the Python standard library."""

import argparse
import hashlib
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Compare an asset directory with the recorded manifest; never modify it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(Path(__file__).with_name("assets_manifest.json").read_text())
    errors = 0
    for name, expected in manifest["files"].items():
        path = args.root / name
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        valid = (
            path.is_file()
            and path.stat().st_size == expected["size_bytes"]
            and digest.hexdigest() == expected["sha256"]
        )
        print(f"{'OK' if valid else 'MISSING OR MISMATCH'} {name}")
        errors += not valid
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
