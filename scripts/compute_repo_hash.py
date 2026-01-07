#!/usr/bin/env python3
"""
Compute content hash of a git repository.

Hashes all tracked files (respects .gitignore) in sorted order.
"""

import hashlib
import subprocess
import sys
from pathlib import Path


def repo_hash(repo_path: Path) -> str:
    """
    Compute SHA3-256 hash of all tracked files in a repo.

    Uses git ls-files to respect .gitignore.
    Files are hashed in sorted order for determinism.
    Each file contributes: relative_path + file_contents
    """
    # Get list of tracked files
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    files = sorted(result.stdout.strip().split("\n"))

    hasher = hashlib.sha3_256()

    for rel_path in files:
        if not rel_path:
            continue

        file_path = repo_path / rel_path
        if not file_path.is_file():
            continue

        # Include path in hash
        hasher.update(rel_path.encode())

        # Include contents
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)

    return hasher.hexdigest()


def main():
    if len(sys.argv) > 1:
        repo_path = Path(sys.argv[1])
    else:
        repo_path = Path.cwd()

    h = repo_hash(repo_path)
    print(f"Repository: {repo_path}")
    print(f"Hash: {h}")
    return h


if __name__ == "__main__":
    main()
