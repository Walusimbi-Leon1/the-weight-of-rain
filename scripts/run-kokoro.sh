#!/usr/bin/env bash
# run-kokoro.sh — generate a single MP3 audiobook from a markdown file using Kokoro-82M.
# Usage: run-kokoro.sh <INPUT.md> <OUTPUT.mp3> <VOICE> <ATEMPO> <BITRATE>
#
# This script runs ON THE KAMATERA AUDIOBOOK VPS (inside /opt/kokoro-env).
# Called remotely by the GitHub Actions workflow `audiobook.yml`.
#
# Recipe (matches SGSS salt-and-silk v1.0-audiobook):
#   Voice: af_heart (warm, expressive American female)
#   Speed: 15% slower (atempo=0.85 via ffmpeg)
#   Quality: 128kbps MP3
#
# Env expected on the VPS:
#   KOKORO_VENV=/opt/kokoro-env   (activate this first)
set -euo pipefail

INPUT="$1"; OUTPUT="$2"; VOICE="${3:-af_heart}"; ATEMPO="${4:-0.85}"; BITRATE="${5:-128k}"
VENV="${KOKORO_VENV:-/opt/kokoro-env}"

mkdir -p "$(dirname "$OUTPUT")"

source "$VENV/bin/activate"

TMP_WAV="$(mktemp --suffix=.wav)"
TMP_TXT="$(mktemp --suffix=.txt)"

# Strip markdown to plain text, keep chapter breaks
python3 - "$INPUT" "$TMP_TXT" <<'PY'
import sys, re, pathlib
md = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
# remove markdown artifacts but keep chapter structure as pauses
text = md
for pat in [r'!\[[^\]]*\]\([^)]*\)',   # images
            r'\[([^\]]+)\]\([^)]+\)',     # links -> anchor text
            r'[*_`#>~\-]',                 # md symbols (keep minimal)
            ]:
    text = re.sub(pat, r'\1' if pat.startswith(r'\[') else '', text)
# Split into chapters so we can concatenate with pauses
parts = re.split(r'\n(?=#+\s)', text)
lines=[]
for i,p in enumerate(parts):
    p=p.strip()
    if not p: continue
    if i>0 and p.startswith('#'):
        p='[pause]\n'+p  # marker we convert to a short silence
    lines.append(p)
open(sys.argv[2],'w',encoding='utf-8').write('\n\n'.join(lines))
PY

# Render with kokoro (kokoro-python / TTS) -> 24kHz wav
python3 - "$TMP_TXT" "$TMP_WAV" "$VOICE" <<'PY'
import sys, torch
from kokoro import KModel, KPipeline
from kokoro.vocoder import KModel as _V   # alias path may vary
from kokoro import KPipeline
inp, out, voice = sys.argv[1:4]
text = open(inp, encoding="utf-8").read()
pipeline = KPipeline(lang="a", model="kokoro", voice=voice, device="cpu", use_zero_sp=True)
# naive: generate whole file as one utterance (good enough for chapter-sized blocks)
# Split on [pause] markers into segments with 1.0s gaps
import re
segs = re.split(r'\[pause\]\s*', text.strip())
import numpy as np, soundfile as sf
chunks=[]
for s in segs:
    if not s.strip(): continue
    segs2 = re.split(r'\n{2,}', s)  # paragraphs
    for chunk in segs2:
        chunk=chunk.strip()
        if not chunk: continue
        for ws in pipeline(chunk):
            audio = ws.audio  # np.ndarray (T,) float32
            chunks.append(audio)
            chunks.append(np.zeros(int(24000*1.0)))  # 1s silent gap between paragraphs
        chunks.append(np.zeros(int(24000*1.5)))       # 1.5s gap between segments
if not chunks:
    raise SystemExit("no audio generated")
audio = np.concatenate(chunks)
sf.write(out, audio, 24000, format='WAV')
print(f"rendered wav: {len(audio)/24000:.1f}s")
PY

# 128kbps MP3, 15% slower (atempo=0.85)
ffmpeg -y -i "$TMP_WAV" -af "atempo=$ATEMPO" -ar 44100 -b:a "$BITRATE" "$OUTPUT" >/dev/null 2>&1

rm -f "$TMP_WAV" "$TMP_TXT"
echo "✅ $OUTPUT"
