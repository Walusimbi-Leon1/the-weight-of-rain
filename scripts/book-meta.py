#!/usr/bin/env python3
"""Emit the book title from book.config.json for the audiobook workflow.
Reads book.config.json (present in every book repo) and prints title=… to stdout
so the workflow can capture it via steps.meta.outputs.title."""
import json, os, sys

cfg_path = os.environ.get("BOOK_CONFIG", "book.config.json")
if not os.path.exists(cfg_path):
    # v2.0 fallback path
    sys.stderr.write(f"# warn: {cfg_path} not found, falling back to dir name\n")
    title = os.path.basename(os.getcwd()).replace("-", " ").title()
    print(title)
else:
    d = json.load(open(cfg_path))
    print(d.get("title", d.get("subtitle", "Untitled")).strip())
