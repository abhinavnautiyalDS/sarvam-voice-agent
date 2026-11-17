"""Owner-side CLI for ingesting UTF-8 text/Markdown files; no public upload API."""
import argparse
from pathlib import Path

from backend.knowledge_base import ingest_text

parser = argparse.ArgumentParser(description="Add a text or Markdown file to AURA Knowledge Base")
parser.add_argument("file", type=Path)
parser.add_argument("--title")
args = parser.parse_args()
source = args.file.expanduser().resolve()
if source.suffix.lower() not in {".txt", ".md"}:
    raise SystemExit("Only .txt and .md files are supported.")
result = ingest_text(args.title or source.stem, source.read_text(encoding="utf-8"), source.name)
print(result)
