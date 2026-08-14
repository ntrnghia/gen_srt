# gen_srt

Generate Vietnamese `.srt` subtitle files from video files in any language using [OpenRouter](https://openrouter.ai) APIs. Source language is auto-detected.

## How it works

1. **Extract & split audio** — ffmpeg extracts mono 16 kHz audio, split into 120-second MP3 chunks
2. **Transcribe** — `fish-audio/transcribe-1` transcribes each chunk with per-character word timestamps, auto-detecting the source language (parallel, 16 workers)
3. **Segment** — words grouped into subtitle segments by start-to-start gap threshold (0.8s)
4. **Translate** — `deepseek/deepseek-v4-flash-0731` auto-detects source language and translates to Vietnamese in batches of 30 (parallel, 16 workers)
5. **Write SRT** — timestamps from transcription, text from translation

## Usage

```bash
pip install requests
export OPENROUTER_API_KEY=sk-or-...
python gen_srt.py "<video_path>"
```

Output: `<video_stem>.srt` next to the input file. Temp files in `%TEMP%/gen_srt_<video_stem>/`.

## Config

| Constant | Default | Description |
|---|---|---|
| `STT_MODEL` | `fish-audio/transcribe-1` | Transcription model (per-character timestamps) |
| `TRANSLATE_MODEL` | `deepseek/deepseek-v4-flash-0731` | Translation model |
| `TRANSLATE_PROVIDER` | `{"order": ["DeepSeek"]}` | OpenRouter provider routing |
| `CHUNK_SEC` | `120` | Audio chunk length in seconds |
| `GAP_THRESHOLD` | `0.8` | Start-to-start gap (seconds) to split segments |
| `MAX_WORKERS` | `16` | Parallel workers for transcription & translation |
| `TRANSLATE_BATCH` | `30` | Subtitle entries per translation request |

## Requirements

- Python 3.12+
- `requests`
- `ffmpeg` on PATH
- OpenRouter API key (`OPENROUTER_API_KEY` env var)

## Cost (~30 min video)

| Step | Model | Cost |
|---|---|---|
| Transcription | `fish-audio/transcribe-1` | ~$0.08 |
| Translation | `deepseek/deepseek-v4-flash-0731` | ~$0.03 |
| **Total** | | **~$0.11** |

## Notes

- `strip_leading_silence()` is available but disabled — `fish-audio/transcribe-1` handles leading silence natively. Enable it if switching to a model with VAD sensitivity (e.g. `x-ai/grok-stt-1.0`).
- `start-to-start` gap segmentation is preferred over `end-to-start` — it's immune to inflated word durations that some models produce across silence.
