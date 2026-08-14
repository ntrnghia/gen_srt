#!/usr/bin/env python3
"""Generate a Vietnamese .srt subtitle file from a video via OpenRouter.

Single-pass transcription with fish-audio/transcribe-1 (word timestamps),
segmented by start-to-start gaps, translated to Vietnamese via deepseek.
Source language is auto-detected for both transcription and translation.

Usage: gen_srt.py <video_path>
Output: <video_path>.srt in the same folder as the video.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple, cast

import requests

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SEC = 120
SAMPLE_RATE = 16000
BITRATE = "32k"

STT_MODEL = "fish-audio/transcribe-1"
TRANSLATE_MODEL = "deepseek/deepseek-v4-flash-0731"
TRANSLATE_PROVIDER: dict[str, Any] = {"order": ["DeepSeek"]}
GAP_THRESHOLD = 0.8
SILENCE_THRESHOLD = -13.0  # dB — speech is ~-10dB, background music ~-15dB+
SILENCE_MIN_DUR = 0.5      # seconds — minimum "silence" duration to detect
MAX_WORKERS = 8
TRANSLATE_BATCH = 30

KEY = os.environ["OPENROUTER_API_KEY"]
STT_API = "https://openrouter.ai/api/v1/audio/transcriptions"
CHAT_API = "https://openrouter.ai/api/v1/chat/completions"


# ── Types ───────────────────────────────────────────────────────────────────
class Word(NamedTuple):
    word: str
    start: float
    end: float


class Segment(NamedTuple):
    start: float
    end: float
    text: str


# ── Utilities ───────────────────────────────────────────────────────────────
def fmt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def extract_and_split(video: str, tmp_dir: str) -> list[str]:
    """Extract audio from *video*, split into CHUNK_SEC mp3 files in *tmp_dir*."""
    if os.path.isdir(tmp_dir):
        existing = sorted(
            f for f in os.listdir(tmp_dir)
            if f.startswith("chunk_") and f.endswith(".mp3")
        )
        if existing:
            print(f"  reusing {len(existing)} existing chunks", flush=True)
            return existing
    else:
        os.makedirs(tmp_dir)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video, "-vn", "-ac", "1",
            "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame", "-b:a", BITRATE,
            "-f", "segment", "-segment_time", str(CHUNK_SEC),
            os.path.join(tmp_dir, "chunk_%03d.mp3"),
        ],
        check=True,
    )
    return sorted(
        f for f in os.listdir(tmp_dir)
        if f.startswith("chunk_") and f.endswith(".mp3")
    )


# ── Transcription ───────────────────────────────────────────────────────────
def segment_words(words: list[Word], gap: float = GAP_THRESHOLD) -> list[Segment]:
    """Group words into segments, breaking on start-to-start gaps > *gap* seconds."""
    segs: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and w.start - cur[-1].start > gap:
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    out: list[Segment] = []
    for s in segs:
        text = "".join(w.word for w in s).strip()
        if text:
            out.append(Segment(s[0].start, s[-1].end, text))
    return out


def _parse_words(raw_words: list[dict[str, Any]]) -> list[Word]:
    out: list[Word] = []
    for w in raw_words:
        word = w.get("word", "")
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)
        if isinstance(word, str) and isinstance(start, (int, float)) and isinstance(end, (int, float)):
            out.append(Word(word, float(start), float(end)))
    return out


def strip_leading_silence(path: str, tmp_dir: str) -> tuple[str, float]:
    """Strip leading non-speech audio from *path*.

    Returns (trimmed_audio_path, offset_seconds). If no leading silence is found,
    returns the original path with offset 0.

    Only trims silence that starts at the very beginning of the file (≥3s), so
    trailing or mid-chunk silence is never mistaken for leading silence.
    """
    r = subprocess.run(
        [
            "ffmpeg", "-i", path, "-af",
            f"silencedetect=noise={SILENCE_THRESHOLD}dB:d={SILENCE_MIN_DUR}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    # Leading silence: silence_start at ~0, followed by silence_end (speech begins).
    # Track the first silence_start; only trim if it's at the file's beginning.
    first_start: float | None = None
    for line in r.stderr.splitlines():
        m = re.search(r"silence_start:\s*([\d.]+)", line)
        if m and first_start is None:
            first_start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*([\d.]+)", line)
        if m and first_start is not None and first_start < 0.5:
            offset = float(m.group(1))
            if offset < 3.0:
                return path, 0.0  # leading silence too short to bother
            trimmed = os.path.join(tmp_dir, "trimmed_" + os.path.basename(path))
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", path, "-ss", str(offset),
                    "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-c:a", "libmp3lame", "-b:a", BITRATE, trimmed,
                ],
                capture_output=True, check=True,
            )
            # Sanity: if trim produced a tiny/invalid file, fall back to original
            if os.path.getsize(trimmed) > 1000:
                return trimmed, offset
            os.remove(trimmed)
            return path, 0.0
    return path, 0.0


def transcribe_chunk(idx: int, path: str, tmp_dir: str) -> tuple[list[Segment], float]:
    # Convert to WAV — fish-audio intermittently 503s on MP3 but accepts WAV reliably
    wav = os.path.join(tmp_dir, "wav_" + os.path.basename(path).replace(".mp3", ".wav"))
    if not os.path.exists(wav):
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), wav],
            capture_output=True, check=True,
        )
    for attempt in range(1, 9):
        try:
            with open(wav, "rb") as f:
                r = requests.post(
                    STT_API,
                    headers={"Authorization": f"Bearer {KEY}"},
                    files={"file": (os.path.basename(wav), f, "audio/wav")},
                    data={
                        "model": STT_MODEL,
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "word",
                    },
                    timeout=900,
                )
            r.raise_for_status()
            d: dict[str, Any] = r.json()
            words = _parse_words(d.get("words", []))
            usage = cast(dict[str, Any], d.get("usage", {}))
            return segment_words(words), float(usage.get("cost", 0.0))
        except Exception as e:  # noqa: BLE001
            wait = min(2 ** attempt, 60)  # 2, 4, 8, 16, 32, 60, 60, 60
            print(f"    [{idx}] attempt {attempt}/8: {e!r}  retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"transcription failed for chunk {idx}")


# ── Translation ─────────────────────────────────────────────────────────────
def translate_batch(texts: list[str]) -> tuple[list[str], float]:
    """Translate *texts* to Vietnamese. Returns (translations, cost)."""
    remaining: dict[int, str] = dict(enumerate(texts))
    result: dict[int, str] = {}
    total_cost = 0.0
    for retry in range(6):
        if not remaining:
            break
        src = {str(i): t for i, t in remaining.items()}
        try:
            r = requests.post(
                CHAT_API,
                headers={"Authorization": f"Bearer {KEY}"},
                json={
                    "model": TRANSLATE_MODEL,
                    "provider": TRANSLATE_PROVIDER,
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You translate subtitle lines to Vietnamese. "
                                "Auto-detect the source language of each line. "
                                "Each key in the input JSON is a SEPARATE subtitle line. "
                                "Translate each one INDEPENDENTLY. Do NOT merge lines. "
                                "Return a JSON object with the SAME keys, each mapped to "
                                "its Vietnamese translation.\n"
                                'Example input: {"0": "你好", "1": "Hello"}\n'
                                'Example output: {"0": "Xin chào", "1": "Xin chào"}'
                            ),
                        },
                        {"role": "user", "content": json.dumps(src, ensure_ascii=False)},
                    ],
                },
                timeout=120,
            )
            r.raise_for_status()
            body: dict[str, Any] = r.json()
            usage = cast(dict[str, Any], body.get("usage", {}))
            total_cost += float(usage.get("cost", 0) or 0)
            content = body["choices"][0]["message"]["content"]
            parsed: dict[str, Any] = json.loads(content)
            for k, v in parsed.items():
                i = int(k)
                if i in remaining and isinstance(v, str) and v.strip():
                    result[i] = v.strip()
                    del remaining[i]
        except Exception as e:  # noqa: BLE001
            wait = min(2 ** retry, 60)
            print(f"    translate: {e!r}  retry in {wait}s", flush=True)
            time.sleep(wait)
    for i, t in remaining.items():
        result[i] = t
    return [result[i] for i in range(len(texts))], total_cost


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <video_path>", file=sys.stderr)
        sys.exit(1)
    video = sys.argv[1]
    if not os.path.isfile(video):
        print(f"Error: file not found: {video}", file=sys.stderr)
        sys.exit(1)

    out_srt = os.path.splitext(video)[0] + ".srt"
    # Unique temp subdir based on video filename
    video_stem = os.path.splitext(os.path.basename(video))[0]
    tmp_dir = os.path.join(tempfile.gettempdir(), f"gen_srt_{video_stem}")

    t0 = time.time()
    print("Step 1: extract + split audio", flush=True)
    chunks = extract_and_split(video, tmp_dir)
    n = len(chunks)

    print(
        f"Step 2: transcribe {n} chunks ({STT_MODEL}, {MAX_WORKERS} workers)",
        flush=True,
    )
    results: list[list[Segment]] = [None] * n  # type: ignore[list-item]
    total_stt = 0.0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(transcribe_chunk, i, os.path.join(tmp_dir, c), tmp_dir): i
            for i, c in enumerate(chunks)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            segs, cost = fut.result()
            results[i] = segs
            total_stt += cost
            print(f"  [{i + 1}/{n}] {len(segs)} segments  ${cost:.4f}", flush=True)

    # Assemble segments with chunk offsets (timestamps from transcription only)
    entries: list[Segment] = []
    prev_end = 0.0
    for i, segs in enumerate(results):
        off = i * CHUNK_SEC
        for seg in segs:
            s = max(seg.start + off, prev_end)  # no overlap across chunk boundaries
            e = seg.end + off
            entries.append(Segment(s, e, seg.text))
            prev_end = e
    print(f"  {len(entries)} segments total", flush=True)

    # Translate (text only — timestamps untouched)
    texts = [seg.text for seg in entries]
    batches = [
        texts[i:i + TRANSLATE_BATCH]
        for i in range(0, len(texts), TRANSLATE_BATCH)
    ]
    print(
        f"Step 3: translate {len(texts)} segments ({TRANSLATE_MODEL}, "
        f"{MAX_WORKERS} workers, {TRANSLATE_BATCH}/batch)",
        flush=True,
    )
    translated: list[str] = [""] * len(texts)
    total_tr = 0.0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(translate_batch, b): bi for bi, b in enumerate(batches)}
        for fut in as_completed(futs):
            bi = futs[fut]
            tr, cost = fut.result()
            total_tr += cost
            base = bi * TRANSLATE_BATCH
            for j, t in enumerate(tr):
                translated[base + j] = t
            print(f"  batch {bi + 1}/{len(batches)}  ${cost:.5f}", flush=True)

    # Write SRT — timestamps from step 2, text from step 3
    print("Step 4: write SRT", flush=True)
    with open(out_srt, "w", encoding="utf-8") as f:
        for idx, (seg, vi) in enumerate(zip(entries, translated), 1):
            f.write(f"{idx}\n{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{vi}\n\n")

    elapsed = time.time() - t0
    total = total_stt + total_tr
    print("\n=== Summary ===", flush=True)
    print(f"  Transcription  {STT_MODEL:35s}  ${total_stt:.4f}", flush=True)
    print(f"  Translation    {TRANSLATE_MODEL:35s}  ${total_tr:.4f}", flush=True)
    print(f"  {'Total':71s}  ${total:.4f}", flush=True)
    print(
        f"  Entries: {len(entries)}  Chunks: {n}  Batches: {len(batches)}  "
        f"Elapsed: {elapsed:.1f}s",
        flush=True,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
