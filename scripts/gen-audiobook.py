#!/usr/bin/env python3
"""
gen-audiobook.py — render a markdown book to a 24kHz WAV using Kokoro-82M,
directly on the GitHub Actions runner (no VPS).

Recipe (matches SGSS salt-and-silk v1.0-audiobook):
  Voice: af_heart  (warm, expressive American female, lang 'a')
  Speed: slowed downstream by atempo=0.85 in ffmpeg (this script just renders natural speed)
  Quality: 128kbps MP3 (encoded in the workflow step, not here)

Splits the book into chapters/paragraphs and inserts 1.0s silent gaps between
paragraphs and 1.5s gaps between chapter boundaries for natural pacing.
"""
import argparse, re, sys
from pathlib import Path

def strip_markdown(md: str) -> str:
    """Minimal markdown -> plain text, preserving chapter structure as paragraph breaks."""
    # drop images
    md = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', md)
    # links -> anchor text (no parenthetical URL)
    md = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', md)
    # code blocks
    md = re.sub(r'```[^\n]*\n.*?```', '', md, flags=re.S)
    # inline code
    md = re.sub(r'`[^`]*`', '', md)
    # horizontal rules
    md = re.sub(r'^(#{3,}|[-*_]){3,}', '', md, flags=re.M)
    # bold/italic markers
    md = re.sub(r'[*_]{1,3}', '', md)
    # trailing spaces
    return md

def chapter_split(text: str):
    """Split on top-level headers into chapter blocks. Returns list[str]."""
    parts = re.split(r'\n(?=#+\s)', text.strip())
    return [p.strip() for p in parts if p.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True, help='.wav path')
    ap.add_argument('--voice', default='af_heart')
    args = ap.parse_args()

    md = Path(args.input).read_text(encoding='utf-8')
    text = strip_markdown(md)
    chapters = chapter_split(text)
    if not chapters:
        chapters = [text]

    from kokoro import KPipeline
    import numpy as np
    import soundfile as sf

    pipe = KPipeline(lang='a', model='kokoro', voice=args.voice, device='cpu',
                     use_zero_sp=True)

    SR = 24000
    chunks = []
    for ci, chap in enumerate(chapters):
        paragraphs = [p for p in re.split(r'\n{2,}', chap) if p.strip()]
        if not paragraphs:
            paragraphs = [chap]
        for pi, para in enumerate(paragraphs):
            for ws in pipe(para):
                chunks.append(ws.audio)
                chunks.append(np.zeros(int(SR * 1.0)))  # 1s paragraph gap
        chunks.append(np.zeros(int(SR * 1.5)))  # 1.5s chapter gap

    if not chunks:
        sys.exit('ERROR: kokoro produced no audio')

    audio = np.concatenate(chunks).astype('float32')
    # normalize to -1..1 then trim a couple seconds of tail silence
    audio /= max(1e-6, np.max(np.abs(audio)))
    print(f"rendered: {len(audio)/SR:.1f}s, voice={args.voice}")
    sf.write(args.output, audio, SR, format='WAV')

if __name__ == '__main__':
    main()
