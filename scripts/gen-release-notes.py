#!/usr/bin/env python3
"""gen-release-notes.py — emit the Markdown release body for an audiobook release.
Replicates the style of SGSS salt-and-silk v1.0-audiobook release notes."""
import argparse, textwrap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--title', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--words', default='?')
    ap.add_argument('--chapters', default='?')
    ap.add_argument('--duration', default='?')
    ap.add_argument('--voice', default='af_heart')
    ap.add_argument('--speed', default='0.85')
    ap.add_argument('--bitrate', default='128k')
    a = ap.parse_args()
    print(f"""## Audiobook — {a.title}

TTS-generated audiobook using Kokoro-82M neural voice.

**Engine:** Kokoro-82M (open-weight TTS, 82M parameters)
**Voice:** {a.voice} (warm, expressive American female)
**Speed:** 15% slower (atempo={a.speed} via ffmpeg)
**Quality:** {a.bitrate} MP3
**Duration:** ~{a.duration} minutes
**Content:** Complete book — {a.words} words, {a.chapters} chapters (current snapshot)

> Each audiobook release captures the book at that point on its writing journey.
> As the book grows, new audiobook releases add on (v{a.version} builds on earlier editions).

Generated via GitHub Actions — Kokoro-82M self-contained pipeline (no VPS).
""")

if __name__ == '__main__':
    main()
