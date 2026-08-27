#!/usr/bin/env python3
"""
gen-audiobook.py — render a markdown book to a 24kHz WAV using Kokoro-82M,
directly on the GitHub Actions runner (no VPS).

Recipe (matches SGSS salt-and-silk v1.0-audiobook):
  Voice: af_heart  (warm, expressive American female, lang_code='a')
  Speed: slowed downstream by atempo=0.85 in ffmpeg (natural speed rendered here)
  Quality: 128kbps MP3 (encoded in the workflow step, not here)

Splits the book into chapter/paragraph blocks, inserts 1.0s silent gaps between
paragraphs and 1.5s gaps between chapters for natural pacing.
Audio is yielded by kokoro as torch float tensors at 24000 Hz; we use scipy
to write a 16-bit PCM WAV (no soundfile dependency).
"""
import argparse, re, sys
from pathlib import Path

def strip_markdown(md: str) -> str:
    """Minimal markdown -> plain text, preserving chapter structure."""
    md = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', md)            # images
    md = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', md)         # links -> anchor text
    md = re.sub(r'```[^\n]*\n.*?```', '', md, flags=re.S)    # code blocks
    md = re.sub(r'`[^`]*`', '', md)                          # inline code
    md = re.sub(r'^#{3,}|[-*_]{3,}', '', md, flags=re.M)    # headings/hr
    md = re.sub(r'[*_]{1,3}', '', md)                        # bold/italic markers
    return md

def chapter_split(text: str):
    """Split on top-level headers into chapter blocks."""
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
    from scipy.io.wavfile import write as wav_write

    SR = 24000
    # kokoro API: lang_code (NOT lang), voice='af_heart', device='cpu'
    pipe = KPipeline(lang_code='a', model='kokoro', voice=args.voice,
                     device='cpu', use_zero_sp=True)

    chunks = []
    for ci, chap in enumerate(chapters):
        paragraphs = [p for p in re.split(r'\n{2,}', chap) if p.strip()]
        if not paragraphs:
            paragraphs = [chap]
        for pi, para in enumerate(paragraphs):
            for ws in pipe(para):
                # ws.audio is a @property returning a torch.FloatTensor (24kHz)
                if ws.audio is not None:
                    arr = ws.audio.numpy() if hasattr(ws.audio, 'numpy') else np.asarray(ws.audio)
                    chunks.append(arr.astype('float32'))
                    chunks.append(np.zeros(int(SR * 1.0), dtype='float32'))  # 1s para gap
        chunks.append(np.zeros(int(SR * 1.5), dtype='float32'))  # 1.5s chapter gap

    if not chunks:
        sys.exit('ERROR: kokoro generated no audio')

    audio = np.concatenate(chunks)
    peak = max(1e-6, float(np.max(np.abs(audio))))
    audio = (audio / peak).astype('float32')
    # to 16-bit PCM
    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    wav_write(args.output, SR, pcm)
    print(f"rendered: {len(audio)/SR:.1f}s, voice={args.voice}, chapters={len(chapters)}")

if __name__ == '__main__':
    main()
