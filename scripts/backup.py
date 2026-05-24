#!/usr/bin/env python3
"""
Backup / restore Qdrant collection snapshots.

Usage:
  python scripts/backup.py create              # create snapshot, download to ./backups/
  python scripts/backup.py list                # list existing snapshots
  python scripts/backup.py restore <file>      # restore from a downloaded snapshot file
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "tg_chat")
BACKUP_DIR = Path("backups")


def create_snapshot():
    client = QdrantClient(url=QDRANT_URL)
    print(f"Creating snapshot of '{QDRANT_COLLECTION}'...")
    snap = client.create_snapshot(collection_name=QDRANT_COLLECTION)
    snap_name = snap.name
    print(f"  Created on server: {snap_name}")

    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = BACKUP_DIR / f"{QDRANT_COLLECTION}_{ts}.snapshot"

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/snapshots/{snap_name}"
    print(f"  Downloading to {local_path}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    size_mb = local_path.stat().st_size / 1024 / 1024
    print(f"Done. {local_path} ({size_mb:.1f} MB)")

    try:
        client.delete_snapshot(collection_name=QDRANT_COLLECTION, snapshot_name=snap_name)
        print(f"  Cleaned up server-side snapshot")
    except Exception as e:
        print(f"  [warn] could not clean server snapshot: {e}", file=sys.stderr)


def list_snapshots():
    if not BACKUP_DIR.exists():
        print("No local backups directory yet.")
    else:
        local = sorted(BACKUP_DIR.glob("*.snapshot"))
        print(f"Local backups in {BACKUP_DIR}/:")
        for f in local:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name}  ({size_mb:.1f} MB)")
        if not local:
            print("  (none)")

    try:
        client = QdrantClient(url=QDRANT_URL)
        server_snaps = client.list_snapshots(collection_name=QDRANT_COLLECTION)
        print(f"\nServer-side snapshots in '{QDRANT_COLLECTION}':")
        for s in server_snaps:
            print(f"  {s.name}")
        if not server_snaps:
            print("  (none)")
    except Exception as e:
        print(f"\n[warn] Could not list server snapshots: {e}", file=sys.stderr)


def restore_snapshot(snapshot_file: str):
    path = Path(snapshot_file)
    if not path.exists():
        print(f"Error: file not found: {snapshot_file}", file=sys.stderr)
        sys.exit(1)

    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/snapshots/upload?priority=snapshot"
    print(f"Uploading {path} to {QDRANT_URL}...")
    with open(path, "rb") as f:
        r = requests.post(url, files={"snapshot": f})
    r.raise_for_status()
    print(f"Done. Collection '{QDRANT_COLLECTION}' restored from {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Qdrant snapshot backup/restore")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create", help="Create a snapshot and download it")
    sub.add_parser("list", help="List local and server snapshots")
    p_restore = sub.add_parser("restore", help="Restore from a local snapshot file")
    p_restore.add_argument("file", help="Path to .snapshot file")

    args = parser.parse_args()

    if args.cmd == "create":
        create_snapshot()
    elif args.cmd == "list":
        list_snapshots()
    elif args.cmd == "restore":
        restore_snapshot(args.file)


if __name__ == "__main__":
    main()
