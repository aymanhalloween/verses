"""
Audio processing pipeline:
  1. Convert to 16kHz mono WAV (ffmpeg)
  2. Silence-based pre-segmentation (ffmpeg silencedetect)
  3. Transcribe each segment (faster-whisper with tarteel model)
  4. Classify segments: Quran (keep) vs Fatiha/dhikr/takbeer (remove)
  5. Stitch kept segments into final MP3
"""

import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Lazy-loaded model (loads once, stays in memory)
# ---------------------------------------------------------------------------

_model = None


def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        # Use int8 on CPU for speed; if GPU available, float16 is better
        compute_type = "float16" if os.environ.get("CUDA_VISIBLE_DEVICES") else "int8"
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

        print(f"[pipeline] Loading tarteel whisper model ({device}, {compute_type})...")
        _model = WhisperModel(
            "tarteel-ai/whisper-base-ar-quran",
            device=device,
            compute_type=compute_type,
        )
        print("[pipeline] Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Quran corpus (loaded once)
# ---------------------------------------------------------------------------

_quran_verses: list[dict] = []
_quran_text_set: set[str] = set()
_fatiha_texts: list[str] = []


def _load_corpus():
    global _quran_verses, _quran_text_set, _fatiha_texts
    if _quran_verses:
        return

    corpus_path = Path(__file__).parent / "quran.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        _quran_verses = json.load(f)

    for v in _quran_verses:
        normalized = _normalize_arabic(v["text"])
        _quran_text_set.add(normalized)
        if v["surah"] == 1:
            _fatiha_texts.append(normalized)


def _normalize_arabic(text: str) -> str:
    """Strip diacritics and normalize Arabic text for fuzzy matching."""
    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]", "", text)
    # Normalize alef variants
    text = re.sub(r"[إأآا]", "ا", text)
    # Normalize taa marbuta
    text = text.replace("ة", "ه")
    # Normalize yaa
    text = text.replace("ى", "ي")
    # Remove tatweel
    text = text.replace("\u0640", "")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Known dhikr / salah phrases
# ---------------------------------------------------------------------------

DHIKR_PATTERNS = [
    r"الله\s*اكبر",           # Allahu Akbar
    r"سبحان\s*ربي\s*العظيم",  # Subhana Rabbiyal Azeem
    r"سبحان\s*ربي\s*الاعلي",  # Subhana Rabbiyal A'la
    r"سمع\s*الله\s*لمن\s*حمد", # Sami Allahu liman hamidah
    r"ربنا\s*ولك\s*الحمد",     # Rabbana wa lakal hamd
    r"السلام\s*عليكم",         # Assalamu Alaikum (tasleem)
    r"استغفر\s*الله",          # Astaghfirullah
    r"اللهم\s*صل",            # Allahumma salli
    r"التحيات",               # Tashahhud
    r"اشهد\s*ان\s*لا\s*اله",  # Ash-hadu an la ilaha
]

_dhikr_compiled = [re.compile(p) for p in DHIKR_PATTERNS]


# ---------------------------------------------------------------------------
# Step 1: Convert to WAV
# ---------------------------------------------------------------------------

def convert_to_wav(input_path: Path, output_path: Path):
    """Convert any audio to 16kHz mono WAV."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Step 2: Silence-based segmentation
# ---------------------------------------------------------------------------

def detect_segments(wav_path: Path, min_silence_len: float = 0.7, silence_thresh: int = -35) -> list[tuple[float, float]]:
    """
    Use ffmpeg silencedetect to find silence boundaries,
    then return list of (start, end) for non-silent segments.
    """
    cmd = [
        "ffmpeg", "-i", str(wav_path), "-af",
        f"silencedetect=noise={silence_thresh}dB:d={min_silence_len}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr

    # Parse silence_start and silence_end from ffmpeg output
    silence_starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", output)]
    silence_ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", output)]

    # Get total duration
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", output)
    if duration_match:
        h, m, s, cs = duration_match.groups()
        total_dur = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100
    else:
        # Fallback: use ffprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(wav_path)],
            capture_output=True, text=True,
        )
        total_dur = float(probe.stdout.strip())

    # Build non-silent segments
    segments: list[tuple[float, float]] = []
    prev_end = 0.0

    for ss, se in zip(silence_starts, silence_ends):
        if ss > prev_end + 0.1:  # non-silent chunk exists before this silence
            segments.append((prev_end, ss))
        prev_end = se

    # Last segment after final silence
    if prev_end < total_dur - 0.1:
        segments.append((prev_end, total_dur))

    # If no silence detected, treat entire file as one segment
    if not segments:
        segments = [(0.0, total_dur)]

    return segments


# ---------------------------------------------------------------------------
# Step 3: Transcribe segments
# ---------------------------------------------------------------------------

def transcribe_segment(model, wav_path: Path, start: float, end: float) -> str:
    """Extract a segment from WAV and transcribe it."""
    # Extract segment to temp file
    seg_path = wav_path.parent / f"seg_{start:.1f}_{end:.1f}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-ss", str(start), "-to", str(end),
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(seg_path),
    ]
    subprocess.run(cmd, capture_output=True)

    try:
        segments_iter, info = model.transcribe(
            str(seg_path),
            language="ar",
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments_iter)
        return text.strip()
    finally:
        seg_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Step 4: Classify
# ---------------------------------------------------------------------------

def classify_segment(text: str, remove_fatiha: bool) -> str:
    """
    Classify transcribed text as:
      'quran'  — keep
      'fatiha' — remove if flag set
      'dhikr'  — remove
      'empty'  — remove
    """
    if not text or len(text.strip()) < 3:
        return "empty"

    normalized = _normalize_arabic(text)

    # Check dhikr patterns first (these are short, distinctive phrases)
    for pattern in _dhikr_compiled:
        if pattern.search(normalized):
            return "dhikr"

    # Check Fatiha match
    if remove_fatiha and _is_fatiha(normalized):
        return "fatiha"

    # Check against Quran corpus
    if _is_quran(normalized):
        return "quran"

    # If it's Arabic text but doesn't match Quran, likely dhikr/dua
    # Short unrecognized segments are probably salah components
    if len(normalized) < 20:
        return "dhikr"

    # Longer unrecognized text — check word overlap with Quran corpus
    overlap = _quran_word_overlap(normalized)
    if overlap > 0.4:
        return "quran"

    return "dhikr"


def _is_fatiha(normalized_text: str) -> bool:
    """Check if text matches Al-Fatiha verses."""
    for fatiha_verse in _fatiha_texts:
        # Check if this verse appears in the text
        if fatiha_verse in normalized_text or normalized_text in fatiha_verse:
            return True
    # Also check word overlap with all fatiha verses combined
    fatiha_combined = " ".join(_fatiha_texts)
    words = set(normalized_text.split())
    fatiha_words = set(fatiha_combined.split())
    if not words:
        return False
    overlap = len(words & fatiha_words) / len(words)
    return overlap > 0.6


def _is_quran(normalized_text: str) -> bool:
    """Check if text matches any Quran verse exactly or as substring."""
    for verse_text in _quran_text_set:
        if normalized_text in verse_text or verse_text in normalized_text:
            return True
    return False


def _quran_word_overlap(normalized_text: str) -> float:
    """
    Calculate what fraction of words in the text appear in the Quran corpus.
    High overlap = likely Quran recitation.
    """
    words = set(normalized_text.split())
    if not words:
        return 0.0

    # Build a set of all unique words in the Quran (cached)
    if not hasattr(_quran_word_overlap, "_corpus_words"):
        all_words: set[str] = set()
        for verse_text in _quran_text_set:
            all_words.update(verse_text.split())
        _quran_word_overlap._corpus_words = all_words  # type: ignore

    corpus_words = _quran_word_overlap._corpus_words  # type: ignore
    matching = words & corpus_words
    return len(matching) / len(words)


# ---------------------------------------------------------------------------
# Step 5: Stitch kept segments into MP3
# ---------------------------------------------------------------------------

def stitch_segments(wav_path: Path, segments: list[tuple[float, float]], output_path: Path):
    """Concatenate kept audio segments into a single MP3."""
    if not segments:
        raise RuntimeError("No Quran segments found in the recording.")

    job_dir = wav_path.parent
    list_file = job_dir / "concat_list.txt"
    part_files: list[Path] = []

    for i, (start, end) in enumerate(segments):
        part = job_dir / f"part_{i:04d}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(wav_path),
            "-ss", str(start), "-to", str(end),
            "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(part),
        ]
        subprocess.run(cmd, capture_output=True)
        part_files.append(part)

    # Write concat list
    with open(list_file, "w") as f:
        for part in part_files:
            f.write(f"file '{part}'\n")

    # Concatenate and encode to MP3
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stitching failed: {result.stderr[-500:]}")

    # Clean up temp files
    for part in part_files:
        part.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_audio(
    audio_path: Path,
    job_dir: Path,
    remove_fatiha: bool,
    update_fn: Callable[[str, float], None],
) -> dict:
    """
    Full pipeline. Returns summary dict:
      { segmentsKept, segmentsRemoved, durationSecs, removedSecs }
    """
    _load_corpus()

    # Step 1: Convert
    update_fn("converting", 0.05)
    wav_path = job_dir / "audio.wav"
    convert_to_wav(audio_path, wav_path)
    update_fn("converting", 0.10)

    # Step 2: Segment by silence
    update_fn("transcribing", 0.12)
    raw_segments = detect_segments(wav_path)
    print(f"[pipeline] Found {len(raw_segments)} segments via silence detection")

    # Step 3: Load model and transcribe
    model = get_model()
    update_fn("transcribing", 0.15)

    classified: list[dict] = []
    total_segs = len(raw_segments)

    for i, (start, end) in enumerate(raw_segments):
        # Update progress: transcription takes 15% -> 75%
        pct = 0.15 + (i / total_segs) * 0.60
        update_fn("transcribing", pct)

        text = transcribe_segment(model, wav_path, start, end)
        label = classify_segment(text, remove_fatiha)

        classified.append({
            "start": start,
            "end": end,
            "text": text,
            "label": label,
        })
        print(f"[pipeline] Segment {i+1}/{total_segs}: [{label:>7}] {start:.1f}-{end:.1f}s | {text[:60]}")

    # Step 4: Filter
    update_fn("filtering", 0.78)
    kept = [(s["start"], s["end"]) for s in classified if s["label"] == "quran"]
    removed = [s for s in classified if s["label"] != "quran"]

    if not kept:
        raise RuntimeError(
            "Could not identify any Quran recitation segments. "
            "The audio may be too short or not contain clear recitation."
        )

    # Step 5: Stitch
    update_fn("stitching", 0.82)
    output_path = job_dir / "output.mp3"
    stitch_segments(wav_path, kept, output_path)
    update_fn("stitching", 0.95)

    # Calculate stats
    kept_duration = sum(e - s for s, e in kept)
    removed_duration = sum(s["end"] - s["start"] for s in removed)

    # Clean up WAV (keep MP3)
    wav_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)

    return {
        "segmentsKept": len(kept),
        "segmentsRemoved": len(removed),
        "durationSecs": round(kept_duration, 1),
        "removedSecs": round(removed_duration, 1),
    }
