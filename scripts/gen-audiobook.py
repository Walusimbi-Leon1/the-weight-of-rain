#!/usr/bin/env python3
"""
gen-audiobook.py — render a markdown book to a 24kHz WAV using Kokoro-82M,
directly on the GitHub Actions runner (no VPS).

Recipe (matches SGSS salt-and-silk v1.0-audiobook):
  Voice: af_heart  (warm, expressive American female, lang_code='a')
  Speed: natural render here; slowed to 85% downstream by ffmpeg atempo=0.85
  Quality: 128kbps MP3 (encoded in the workflow step)

Note on the kokoro 0.9.4 API (verified):
  - KPipeline(lang_code='a')                   # 'a' = American English
  - pipe = KPipeline(lang_code='a')            # default model auto-inits from HF
  - for ws in pipe(text, voice='af_heart'):    # voice passed at CALL time
        ws.audio  -> torch.FloatTensor (24kHz)  # .audio is a @property
Audio saved via scipy (no soundfile needed).
"""
import argparse, re, sys
from pathlib import Path

def strip_markdown(md: str) -> str:
    md = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', md)            # images
    md = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', md)         # links -> anchor text
    md = re.sub(r'```[^\n]*\n.*?```', '', md, flags=re.S)    # code blocks
    md = re.sub(r'`[^`]*`', '', md)                          # inline code
    md = re.sub(r'^#{3,}|[-*_]{3,}', '', md, flags=re.M)    # headings/hr
    md = re.sub(r'[*_]{1,3}', '', md)                        # bold/italic markers
    return md

def chapter_split(text: str):
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
    pipe = KPipeline(lang_code='a', device='cpu')  # auto-init Kokoro-82M from HF

    chunks = []
    for ci, chap in enumerate(chapters):
        paragraphs = [p for p in re.split(r'\n{2,}', chap) if p.strip()]
        if not paragraphs:
            paragraphs = [chap]
        for pi, para in enumerate(paragraphs):
            for ws in pipe(para, voice=args.voice, speed=1.0):
                # ws.audio is a @property -> torch.FloatTensor at 24kHz
                if ws.audio is None:
                    continue
                arr = ws.audio.numpy() if hasattr(ws.audio, 'numpy') else np.asarray(ws.audio)
                chunks.append(arr.astype('float32'))
                chunks.append(np.zeros(int(SR * 1.0), dtype='float32'))  # 1s para gap
        chunks.append(np.zeros(int(SR * 1.5), dtype='float32'))  # 1.5s chapter gap

    if not chunks:
        sys.exit('ERROR: kokoro generated no audio')

    audio = np.concatenate(chunks)
    peak = max(1e-6, float(np.max(np.abs(audio))))
    audio = (audio / peak).astype('float32')
    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    wav_write(args.output, SR, pcm)
    print(f"rendered: {len(audio)/SR:.1f}s, voice={args.voice}, chapters={len(chapters)}")

if __name__ == '__main__':
    main()
