"""
Audio processing pipeline for Taraweeh recitation.

Strategy: structure first, transcription only as a sanity check.

  1. Convert to 16kHz mono WAV.
  2. One silence pass → skeleton of pauses.
  3. Invert → speech segments (gaps between silences).
  4. Classify by duration + position:
       - short  → dhikr/takbeer/dua     (drop)
       - medium → confirm with one transcript slice
       - long   → Quran recitation      (keep, split off Fatiha)
  5. Fatiha boundary = first silence ≥ 0.7s in the 20–50s window
     of a recitation segment. That's the "ameen" pause.
  6. Trim trailing takbeer: last silence > 0.5s inside final 10s.
  7. Stitch with a 30ms crossfade to avoid clicks.
"""

import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Config (one place, no cascades)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
SILENCE_THRESH_DB = -30          # silencedetect noise floor
SILENCE_MIN_SEC = 0.4            # silences shorter than this are ignored
MERGE_GAP_SEC = 1.8              # merge segments separated by < this (keeps
                                 # Fatiha+Surah together across the ameen pause)
FATIHA_GAP_MIN_SEC = 0.7         # ameen-pause must be at least this long
FATIHA_SEARCH_START = 15.0       # earliest position of ameen pause in a block
FATIHA_SEARCH_END = 55.0         # latest position
TAKBEER_TAIL_SEC = 10.0          # look for takbeer in the last 10s
TAKBEER_MAX_TRIM_SEC = 4.0       # don't trim more than this
SHORT_SEG_SEC = 6.0              # below this = dhikr, drop
LONG_SEG_SEC = 25.0              # above this = recitation, no transcript needed
CROSSFADE_MS = 30                # stitch crossfade

# Fatiha confirmation markers. One hit is enough; these are robust against
# noisy Whisper output on fast congregational Fatiha.
FATIHA_MARKERS = ("الحمد", "العالمين", "الرحمن", "المستقيم",
                  "المغضوب", "الضالين", "نعبد", "نستعين")

# Dhikr markers (for classifying medium-length segments).
DHIKR_MARKERS = ("الله اكبر", "سبحان ربي", "ربنا ولك", "سمع الله",
                 "السلام عليكم", "استغفر الله", "التحيات", "اللهم صل")


# ---------------------------------------------------------------------------
# Whisper model (lazy)
# ---------------------------------------------------------------------------

_model = None
_processor = None


def get_model():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_id = os.environ.get("WHISPER_MODEL", "tarteel-ai/whisper-base-ar-quran")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[pipeline] Loading {model_id} on {device}...")
        _processor = WhisperProcessor.from_pretrained(model_id)
        _model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        _model.eval()
    return _model, _processor


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def convert_to_wav(src: Path, dst: Path) -> None:
    r = _run(["ffmpeg", "-y", "-i", str(src),
              "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(dst)])
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg convert failed: {r.stderr[-500:]}")


def get_duration(wav: Path) -> float:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", str(wav)])
    return float(r.stdout.strip())


def detect_silences(wav: Path) -> list[tuple[float, float]]:
    """All silences ≥ SILENCE_MIN_SEC. Returns (start, end) in seconds."""
    r = _run(["ffmpeg", "-i", str(wav), "-af",
              f"silencedetect=noise={SILENCE_THRESH_DB}dB:d={SILENCE_MIN_SEC}",
              "-f", "null", "-"])
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", r.stderr)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", r.stderr)]
    return list(zip(starts, ends[:len(starts)]))


def speech_segments(silences: list[tuple[float, float]], total: float
                    ) -> list[tuple[float, float]]:
    """Gaps between silences = speech segments. Adjacent segments separated
    by a gap shorter than MERGE_GAP_SEC are merged — this keeps Fatiha and
    its surah together across the short ameen pause, and keeps ayah-breaths
    from fragmenting a recitation."""
    raw: list[tuple[float, float]] = []
    prev = 0.0
    for ss, se in silences:
        if ss > prev + 0.05:
            raw.append((prev, ss))
        prev = se
    if prev < total - 0.05:
        raw.append((prev, total))

    if not raw:
        return raw
    merged: list[tuple[float, float]] = [raw[0]]
    for s, e in raw[1:]:
        ps, pe = merged[-1]
        if s - pe < MERGE_GAP_SEC:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Transcription (fixed 5s confirmation slice)
# ---------------------------------------------------------------------------

def _load_wav_slice(wav: Path, start: float, end: float) -> np.ndarray:
    tmp = wav.parent / f"_slice_{start:.1f}_{end:.1f}.wav"
    _run(["ffmpeg", "-y", "-i", str(wav), "-ss", str(start), "-to", str(end),
          "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", str(tmp)])
    try:
        with wave.open(str(tmp), "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            return data.astype(np.float32) / 32768.0
    finally:
        tmp.unlink(missing_ok=True)


def transcribe(wav: Path, start: float, end: float) -> str:
    import torch
    model, processor = get_model()
    audio = _load_wav_slice(wav, start, end)
    if len(audio) < 1000:
        return ""
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    feats = inputs.input_features.to(next(model.parameters()).device)
    with torch.no_grad():
        ids = model.generate(feats, max_new_tokens=220)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(m in text for m in markers)


# ---------------------------------------------------------------------------
# Fatiha boundary + takbeer trim
# ---------------------------------------------------------------------------

def find_fatiha_end(seg_start: float, seg_end: float,
                    silences: list[tuple[float, float]]) -> float | None:
    """First silence ≥ FATIHA_GAP_MIN_SEC inside the search window — that's
    the ameen pause. Returns None if no such pause found (we'd rather keep
    too much than chop real recitation)."""
    lo = seg_start + FATIHA_SEARCH_START
    hi = min(seg_start + FATIHA_SEARCH_END, seg_end)
    for ss, se in silences:
        if lo <= ss <= hi and (se - ss) >= FATIHA_GAP_MIN_SEC:
            return se
    return None


def trim_takbeer(seg_start: float, seg_end: float,
                 silences: list[tuple[float, float]]) -> float:
    """Last silence >0.5s in final TAKBEER_TAIL_SEC. Only trim if the
    resulting chop is ≤ TAKBEER_MAX_TRIM_SEC."""
    tail_lo = seg_end - TAKBEER_TAIL_SEC
    candidates = [ss for ss, se in silences
                  if tail_lo <= ss < seg_end and (se - ss) >= 0.5]
    if not candidates:
        return seg_end
    cut = candidates[-1]
    return cut if (seg_end - cut) <= TAKBEER_MAX_TRIM_SEC else seg_end


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_segment(wav: Path, start: float, end: float,
                     silences: list[tuple[float, float]],
                     remove_fatiha: bool) -> list[dict]:
    """Return a list of {start, end, label} sub-segments.
    Label is 'quran' (keep) or 'dhikr' (drop)."""
    dur = end - start

    if dur < SHORT_SEG_SEC:
        return [{"start": start, "end": end, "label": "dhikr"}]

    if dur <= LONG_SEG_SEC:
        # Medium: transcribe a 5s slice from the middle and decide.
        mid = start + dur / 2
        text = transcribe(wav, max(start, mid - 2.5), min(end, mid + 2.5))
        print(f"[pipeline]   mid-slice text: {text[:80]!r}")
        if contains_any(text, DHIKR_MARKERS) or len(text) < 8:
            return [{"start": start, "end": end, "label": "dhikr"}]
        return [{"start": start, "end": end, "label": "quran"}]

    # Long: Quran recitation. Split off Fatiha + trim trailing takbeer.
    keep_end = trim_takbeer(start, end, silences)
    if keep_end <= start:
        return [{"start": start, "end": end, "label": "dhikr"}]

    if not remove_fatiha:
        return [{"start": start, "end": keep_end, "label": "quran"}]

    cut = find_fatiha_end(start, keep_end, silences)
    if cut is None:
        # No ameen-shaped pause in the window. Could mean: imam ran Fatiha
        # straight into the surah, or this block was surah-only. Keep it all
        # rather than blindly chopping 30s.
        print(f"[pipeline]   no ameen pause in [{start+FATIHA_SEARCH_START:.1f}, "
              f"{min(start+FATIHA_SEARCH_END, keep_end):.1f}]s — keeping whole block")
        return [{"start": start, "end": keep_end, "label": "quran"}]

    # Sanity-check the 5s before the cut — should look like Fatiha.
    text = transcribe(wav, max(start, cut - 5.0), cut)
    print(f"[pipeline]   fatiha-check text: {text[:80]!r}")
    if not contains_any(text, FATIHA_MARKERS):
        print("[pipeline]   (no fatiha marker hit — using silence boundary anyway)")

    parts = []
    if cut > start + 1.0:
        parts.append({"start": start, "end": cut, "label": "fatiha"})
    if keep_end > cut + 1.0:
        parts.append({"start": cut, "end": keep_end, "label": "quran"})
    return parts


# ---------------------------------------------------------------------------
# Stitch with crossfade
# ---------------------------------------------------------------------------

def stitch(wav: Path, segments: list[tuple[float, float]], out: Path) -> None:
    if not segments:
        raise RuntimeError("No Quran segments found in the recording.")

    job = wav.parent
    parts: list[Path] = []
    for i, (s, e) in enumerate(segments):
        p = job / f"part_{i:04d}.wav"
        _run(["ffmpeg", "-y", "-i", str(wav), "-ss", str(s), "-to", str(e),
              "-c:a", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", str(p)])
        parts.append(p)

    try:
        if len(parts) == 1:
            r = _run(["ffmpeg", "-y", "-i", str(parts[0]),
                      "-c:a", "libmp3lame", "-b:a", "192k", str(out)])
        else:
            # Chain acrossfade across all parts.
            inputs: list[str] = []
            for p in parts:
                inputs += ["-i", str(p)]
            xf = CROSSFADE_MS / 1000.0
            filters = []
            cur = "[0:a]"
            for i in range(1, len(parts)):
                nxt = f"[{i}:a]"
                label = f"[x{i}]"
                filters.append(f"{cur}{nxt}acrossfade=d={xf}:c1=tri:c2=tri{label}")
                cur = label
            filter_complex = ";".join(filters)
            r = _run(["ffmpeg", "-y", *inputs,
                      "-filter_complex", filter_complex,
                      "-map", cur, "-c:a", "libmp3lame", "-b:a", "192k",
                      str(out)])
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg stitch failed: {r.stderr[-500:]}")
    finally:
        for p in parts:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_audio(
    audio_path: Path,
    job_dir: Path,
    remove_fatiha: bool,
    update_fn: Callable[[str, float], None],
) -> dict:
    update_fn("converting", 0.05)
    wav = job_dir / "audio.wav"
    convert_to_wav(audio_path, wav)

    update_fn("transcribing", 0.15)
    total = get_duration(wav)
    silences = detect_silences(wav)
    segs = speech_segments(silences, total)
    print(f"[pipeline] {total:.1f}s audio, {len(silences)} silences, {len(segs)} speech segments")
    for s, e in segs:
        print(f"[pipeline]   seg {s:.1f}–{e:.1f}s ({e-s:.1f}s)")

    update_fn("transcribing", 0.25)
    classified: list[dict] = []
    for i, (s, e) in enumerate(segs):
        update_fn("transcribing", 0.25 + 0.55 * (i / max(1, len(segs))))
        parts = classify_segment(wav, s, e, silences, remove_fatiha)
        for p in parts:
            classified.append(p)
            tag = "+" if p["label"] == "quran" else "-"
            print(f"[pipeline]   {tag} [{p['label']:>6}] {p['start']:.1f}–{p['end']:.1f}s")

    update_fn("filtering", 0.82)
    kept = [(p["start"], p["end"]) for p in classified if p["label"] == "quran"]
    removed = [p for p in classified if p["label"] != "quran"]
    if not kept:
        raise RuntimeError("Could not identify any Quran recitation segments.")

    update_fn("stitching", 0.88)
    out = job_dir / "output.mp3"
    stitch(wav, kept, out)
    update_fn("stitching", 0.98)

    kept_dur = sum(e - s for s, e in kept)
    removed_dur = sum(p["end"] - p["start"] for p in removed)
    wav.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)

    return {
        "segmentsKept": len(kept),
        "segmentsRemoved": len(removed),
        "durationSecs": round(kept_dur, 1),
        "removedSecs": round(removed_dur, 1),
    }
