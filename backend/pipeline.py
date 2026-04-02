"""
Audio processing pipeline:
  1. Convert to 16kHz mono WAV
  2. Find structural boundaries (ruku/sujood silences >3s)
  3. Each block between boundaries = one continuous recitation
  4. Transcribe in windows to find where Fatiha ends
  5. Use fine-grained silence detection to find exact cut points
  6. Trim ruku takbeer from block ends
  7. Keep Surah portions, remove Fatiha/dhikr
  8. Export as MP3

Key principles:
  - NEVER cut within continuous recitation (no choppy audio)
  - Only cut at natural silence boundaries
  - Use transcription to approximate Fatiha→Surah zone, then silence
    detection to find the exact pause after ameen
"""

import json
import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Lazy-loaded model
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

        print(f"[pipeline] Loading model '{model_id}' on {device}...")
        _processor = WhisperProcessor.from_pretrained(model_id)
        _model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        _model.eval()
        print("[pipeline] Model loaded.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Quran corpus
# ---------------------------------------------------------------------------

_quran_verses: list[dict] = []
_fatiha_texts: list[str] = []
_fatiha_combined: str = ""


def _load_corpus():
    global _quran_verses, _fatiha_texts, _fatiha_combined
    if _quran_verses:
        return

    corpus_path = Path(__file__).parent / "quran.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        _quran_verses = json.load(f)

    for v in _quran_verses:
        if v["surah"] == 1:
            _fatiha_texts.append(_normalize_arabic(v["text"]))

    _fatiha_combined = " ".join(_fatiha_texts)


def _normalize_arabic(text: str) -> str:
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC"
        r"\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]",
        "", text,
    )
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("\u0640", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Fatiha ending markers — last ayah or nearby text (fuzzy: shorter substrings)
FATIHA_END_MARKERS = [
    "الضالين",
    "المغضوب عليهم",
    "المغضوب",
    "بعليهم ولا",  # common model garble for "عليهم ولا"
    "عليهم ولا",
]

# Fatiha markers — shorter substrings to survive noisy transcription
# The Quran model produces garbled text for Fatiha (spoken fast by congregation)
FATIHA_MARKERS = [
    "الحمد لله",
    "العالمين",
    "الرحمن",
    "الرحيم",
    "يوم الدين",
    "نعبد",
    "نستعين",
    "الصراط",
    "المستقيم",
    "صراط الذين",
    "صراط",
    "انعمت",
    "المغضوب",
    "الضالين",
]

DHIKR_PATTERNS = [
    r"الله?\s*اكبر",
    r"سبحان\s*ربي",
    r"سمع\s*الله",
    r"ربنا\s*و?لك\s*الحمد",
    r"سلا[مو]\s*عل[يى]كم",
    r"استغفر\s*الله",
    r"اللهم\s*صل",
    r"التحيات",
    r"اشهد\s*ان\s*لا",
    r"سلام\s*علكم",
    r"سلو?لكم",
    r"الله\s*اكر",
]

_dhikr_compiled = [re.compile(p) for p in DHIKR_PATTERNS]


# ---------------------------------------------------------------------------
# Step 1: Convert to WAV
# ---------------------------------------------------------------------------

def convert_to_wav(input_path: Path, output_path: Path):
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr[-500:]}")


# ---------------------------------------------------------------------------
# Step 2: Find structural blocks (only split at long silences)
# ---------------------------------------------------------------------------

def _get_duration(wav_path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


def find_fine_silences(wav_path: Path, min_dur: float = 0.3,
                       threshold: int = -25) -> list[tuple[float, float]]:
    """
    Find ALL silences in the full WAV file (short pauses, breaths, etc.).
    Returns list of (silence_start, silence_end) with absolute timestamps.
    Used for precise cut-point detection within zones.
    """
    for thresh in [threshold, threshold + 3, threshold + 6, threshold + 9]:
        cmd = [
            "ffmpeg", "-i", str(wav_path), "-af",
            f"silencedetect=noise={thresh}dB:d={min_dur}",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr
        starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", output)]
        ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", output)]
        if starts:
            pairs = list(zip(starts, ends[:len(starts)]))
            return pairs
    return []


def find_structural_blocks(wav_path: Path) -> list[tuple[float, float]]:
    """
    Find continuous recitation blocks by splitting ONLY at long silences (>3s).
    These correspond to ruku, sujood, and transitions between rak'ahs.

    Short pauses (breaths between ayahs) are NOT splits — the audio stays intact.
    """
    total_dur = _get_duration(wav_path)

    # Find silences
    for thresh in [-25, -22, -20, -18]:
        cmd = [
            "ffmpeg", "-i", str(wav_path), "-af",
            f"silencedetect=noise={thresh}dB:d=3.0",
            "-f", "null", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr
        starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", output)]
        ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", output)]
        if len(starts) >= 2:
            print(f"[pipeline] Structural silences at {thresh}dB: {len(starts)} breaks")
            break
    else:
        print("[pipeline] Warning: few structural silences found")
        return [(0.0, total_dur)]

    # Build blocks between structural silences
    blocks: list[tuple[float, float]] = []
    prev_end = 0.0
    for ss, se in zip(starts, ends):
        if ss > prev_end + 3.0:  # block must be > 3s to matter
            blocks.append((prev_end, ss))
        prev_end = se
    if prev_end < total_dur - 3.0:
        blocks.append((prev_end, total_dur))

    # Filter out very short blocks (< 5s) — these are just transition noise
    blocks = [(s, e) for s, e in blocks if e - s >= 5.0]

    for i, (s, e) in enumerate(blocks):
        print(f"[pipeline]   Block {i+1}: {s:.1f}-{e:.1f}s ({e-s:.1f}s)")

    return blocks


# ---------------------------------------------------------------------------
# Step 3: Transcribe windows within a block
# ---------------------------------------------------------------------------

def _load_wav_segment(wav_path: Path, start: float, end: float) -> np.ndarray:
    seg_path = wav_path.parent / f"seg_{start:.1f}_{end:.1f}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(wav_path),
        "-ss", str(start), "-to", str(end),
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(seg_path),
    ]
    subprocess.run(cmd, capture_output=True)
    try:
        with wave.open(str(seg_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        seg_path.unlink(missing_ok=True)


def _transcribe_audio(model, processor, audio: np.ndarray) -> str:
    import torch
    if len(audio) < 100:
        return ""
    device = next(model.parameters()).device
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=440)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def transcribe_block_windows(model_and_proc, wav_path: Path, start: float, end: float,
                              small_windows: bool = False) -> list[dict]:
    """
    Transcribe a block in overlapping windows.
    If small_windows=True, use 14s windows (for precise Fatiha boundary detection).
    Otherwise use 28s windows (for general classification).
    Returns list of { window_start, window_end, text } for each window.
    """
    model, processor = model_and_proc
    duration = end - start
    windows = []

    window_size = 14.0 if small_windows else 28.0
    step = window_size - 2.0  # 2s overlap

    if duration <= window_size:
        audio = _load_wav_segment(wav_path, start, end)
        text = _transcribe_audio(model, processor, audio)
        windows.append({"abs_start": start, "abs_end": end, "text": text})
        return windows

    pos = start
    while pos < end:
        chunk_end = min(pos + window_size, end)
        if chunk_end - pos < 3.0:
            break
        audio = _load_wav_segment(wav_path, pos, chunk_end)
        text = _transcribe_audio(model, processor, audio)
        windows.append({"abs_start": pos, "abs_end": chunk_end, "text": text})
        pos += step

    return windows


# ---------------------------------------------------------------------------
# Step 4: Find Fatiha boundary within a rak'ah block
# ---------------------------------------------------------------------------

def _text_is_fatiha(normalized_text: str) -> bool:
    """Check if text chunk contains Fatiha content.
    Uses fuzzy matching since the model produces noisy transcriptions of Fatiha
    (spoken quickly by congregation, with ameen overlapping).
    """
    hits = sum(1 for m in FATIHA_MARKERS if m in normalized_text)
    if hits >= 1:
        return True

    words = set(normalized_text.split())
    fatiha_words = set(_fatiha_combined.split())
    if not words or not fatiha_words:
        return False

    overlap = len(words & fatiha_words) / len(words)
    return overlap > 0.25


def _text_has_fatiha_end(normalized_text: str) -> bool:
    """Check if text contains the ending of Fatiha."""
    for marker in FATIHA_END_MARKERS:
        if marker in normalized_text:
            return True
    # Also check if text ends with partial "ولا ال" (truncated at window boundary)
    if normalized_text.rstrip().endswith("ولا") or normalized_text.rstrip().endswith("ولا ال"):
        return True
    return False


def _text_is_dhikr(text: str) -> bool:
    """Check if text is dhikr/salah phrases."""
    normalized = _normalize_arabic(text)
    for pattern in _dhikr_compiled:
        if pattern.search(normalized):
            return True
    return len(normalized) < 10


def find_fatiha_end_in_block(windows: list[dict], fine_silences: list[tuple[float, float]]) -> float | None:
    """
    Scan transcription windows to find where Fatiha ends, then use
    fine-grained silence detection to find the EXACT pause after ameen.
    Returns the absolute timestamp where Surah begins (cut point),
    or None if no Fatiha detected.
    """
    if not windows:
        return None

    # Check if the first window even contains Fatiha
    first_text = _normalize_arabic(windows[0]["text"])
    if not _text_is_fatiha(first_text):
        return None  # This block doesn't start with Fatiha

    # Find the window where Fatiha ends (contains "ولا الضالين" or similar)
    fatiha_end_window = None
    for i, w in enumerate(windows):
        normalized = _normalize_arabic(w["text"])
        if _text_has_fatiha_end(normalized):
            fatiha_end_window = w
            break

    # If we couldn't find the end marker, fall back to text-change detection
    if fatiha_end_window is None:
        for i, w in enumerate(windows):
            normalized = _normalize_arabic(w["text"])
            if i > 0 and not _text_is_fatiha(normalized):
                cut_point = w["abs_start"]
                print(f"[pipeline]   Fatiha likely ends around {cut_point:.1f}s (text change)")
                return cut_point
        # All windows are Fatiha
        print("[pipeline]   Entire block appears to be Fatiha")
        return windows[-1]["abs_end"]

    # We know Fatiha ends in fatiha_end_window.
    # The ameen comes right after "ولا الضالين", then there's a pause before surah.
    # Strategy: find the first substantial silence AFTER the midpoint of this window.
    # The midpoint approximation: "ولا الضالين" is likely near the end of the window,
    # so the ameen pause is near or just after the window end.
    search_from = fatiha_end_window["abs_start"] + (fatiha_end_window["abs_end"] - fatiha_end_window["abs_start"]) * 0.4
    zone_end = fatiha_end_window["abs_end"] + 10.0

    # Find silences in this zone, sorted by start time
    zone_silences = sorted(
        [(ss, se) for ss, se in fine_silences
         if ss >= search_from and se <= zone_end and (se - ss) >= 0.3],
        key=lambda x: x[0],
    )

    if zone_silences:
        # Pick the first silence that's at least 0.5s long (the ameen→surah pause)
        # If none are 0.5s, use the first one
        substantial = [s for s in zone_silences if s[1] - s[0] >= 0.5]
        best = substantial[0] if substantial else zone_silences[0]
        cut_point = best[1]  # Cut at silence END (right after the pause)
        print(f"[pipeline]   Fatiha ends: search {search_from:.1f}-{zone_end:.1f}s, "
              f"silence {best[0]:.1f}-{best[1]:.1f}s, cut at {cut_point:.1f}s")
        return cut_point

    # No fine silences found — fall back to window end + small buffer
    cut_point = fatiha_end_window["abs_end"] + 3.0
    print(f"[pipeline]   Fatiha ends around {cut_point:.1f}s (no fine silence found)")
    return cut_point


# ---------------------------------------------------------------------------
# Step 5: Classify each block
# ---------------------------------------------------------------------------

def trim_takbeer_from_end(block_end: float, fine_silences: list[tuple[float, float]]) -> float:
    """
    Find and trim the ruku takbeer ("Allahu Akbar") from the end of a recitation block.
    The takbeer is typically 1.5-3s, preceded by a short pause.
    Look for the last silence in the final ~8s of the block, and cut there.
    """
    search_start = block_end - 8.0
    # Find silences in the tail zone
    tail_silences = [
        (ss, se) for ss, se in fine_silences
        if ss >= search_start and se <= block_end and (se - ss) >= 0.2
    ]

    if tail_silences:
        # Use the LAST silence — it's right before the takbeer
        last_silence = tail_silences[-1]
        # Cut at the start of that silence (before the takbeer)
        trimmed = last_silence[0]
        trim_amount = block_end - trimmed
        if 1.0 <= trim_amount <= 6.0:
            print(f"[pipeline]   Trimming takbeer: {trimmed:.1f}s (was {block_end:.1f}s, cut {trim_amount:.1f}s)")
            return trimmed
        # If the trim amount seems wrong, try the second-to-last silence
        if len(tail_silences) >= 2:
            second_last = tail_silences[-2]
            trimmed2 = second_last[0]
            trim_amount2 = block_end - trimmed2
            if 1.0 <= trim_amount2 <= 6.0:
                print(f"[pipeline]   Trimming takbeer: {trimmed2:.1f}s (was {block_end:.1f}s, cut {trim_amount2:.1f}s)")
                return trimmed2

    # No suitable silence found — trim a conservative 2s
    trimmed = block_end - 2.0
    print(f"[pipeline]   Trimming takbeer (no silence): {trimmed:.1f}s (was {block_end:.1f}s)")
    return trimmed


def classify_block(windows: list[dict], block_start: float, block_end: float,
                   is_first_recitation: bool, remove_fatiha: bool,
                   fine_silences: list[tuple[float, float]]) -> list[dict]:
    """
    Given a block's transcription windows, decide what to keep/remove.

    Returns list of { start, end, label } where label is 'quran' or 'fatiha' or 'dhikr'.
    A block may be split into at most 2 parts: fatiha (remove) + surah (keep).
    """
    duration = block_end - block_start
    all_text = " ".join(w["text"] for w in windows)

    # Short blocks (< 8s) — check if dhikr
    if duration < 8.0:
        if _text_is_dhikr(all_text):
            return [{"start": block_start, "end": block_end, "label": "dhikr"}]

    # Trim ruku takbeer from the end of long recitation blocks (>30s)
    effective_end = block_end
    if duration > 30.0:
        effective_end = trim_takbeer_from_end(block_end, fine_silences)

    # If this is the first recitation block in a rak'ah AND remove_fatiha is on,
    # try to find where Fatiha ends
    if is_first_recitation and remove_fatiha:
        fatiha_end = find_fatiha_end_in_block(windows, fine_silences)
        if fatiha_end is not None:
            result = []
            # Fatiha portion
            if fatiha_end > block_start:
                result.append({"start": block_start, "end": fatiha_end, "label": "fatiha"})
            # Surah portion (if there's anything left)
            if fatiha_end < effective_end:
                result.append({"start": fatiha_end, "end": effective_end, "label": "quran"})
            return result

    # Otherwise, entire block is Quran recitation (with trimmed end)
    return [{"start": block_start, "end": effective_end, "label": "quran"}]


# ---------------------------------------------------------------------------
# Step 6: Stitch
# ---------------------------------------------------------------------------

def stitch_segments(wav_path: Path, segments: list[tuple[float, float]], output_path: Path):
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

    with open(list_file, "w") as f:
        for part in part_files:
            f.write(f"file '{part}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stitching failed: {result.stderr[-500:]}")

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
    _load_corpus()

    # Step 1: Convert
    update_fn("converting", 0.05)
    wav_path = job_dir / "audio.wav"
    convert_to_wav(audio_path, wav_path)
    update_fn("converting", 0.10)

    # Step 2: Find structural blocks
    update_fn("transcribing", 0.12)
    blocks = find_structural_blocks(wav_path)
    total_dur = _get_duration(wav_path)

    if not blocks:
        raise RuntimeError("Could not find any audio content in the recording.")

    # Step 2b: Find fine-grained silences for precise cut points
    update_fn("transcribing", 0.14)
    fine_silences = find_fine_silences(wav_path, min_dur=0.3, threshold=-25)
    print(f"[pipeline] Found {len(fine_silences)} fine-grained silences for precision cuts")

    # Step 3: Load model
    model_and_proc = get_model()
    update_fn("transcribing", 0.18)

    # Step 4: Process each block
    all_segments: list[dict] = []
    is_first_recitation = True  # tracks first recitation block per rak'ah

    for i, (bstart, bend) in enumerate(blocks):
        pct = 0.18 + (i / len(blocks)) * 0.58
        update_fn("transcribing", pct)

        bdur = bend - bstart

        # Very short blocks (< 5s) at start or end are dhikr
        if bdur < 5.0:
            all_segments.append({"start": bstart, "end": bend, "label": "dhikr"})
            print(f"[pipeline] Block {i+1}: [  dhikr] {bstart:.1f}-{bend:.1f}s ({bdur:.1f}s) — too short")
            continue

        # Blocks in the opening (first 30s) are takbeer/opening
        if bstart < 30.0:
            all_segments.append({"start": bstart, "end": bend, "label": "dhikr"})
            print(f"[pipeline] Block {i+1}: [  dhikr] {bstart:.1f}-{bend:.1f}s ({bdur:.1f}s) — opening")
            is_first_recitation = True
            continue

        # Blocks in the last portion (after last long recitation) are closing
        # Detect: if this block starts after a long gap from the previous big block
        if i > 0:
            prev_end = blocks[i-1][1]
            gap = bstart - prev_end
            # If there was a big gap AND this block is short, it's closing
            if gap > 10.0 and bdur < 30.0:
                all_segments.append({"start": bstart, "end": bend, "label": "dhikr"})
                print(f"[pipeline] Block {i+1}: [  dhikr] {bstart:.1f}-{bend:.1f}s ({bdur:.1f}s) — closing/transition")
                continue

        # Transcribe this block in windows
        # For first recitation blocks (containing Fatiha), use small 14s windows
        # in the first 70s for precise Fatiha boundary detection
        print(f"[pipeline] Block {i+1}: Transcribing {bstart:.1f}-{bend:.1f}s ({bdur:.1f}s)...")
        if is_first_recitation and remove_fatiha and bdur > 30.0:
            # Small windows for Fatiha zone (first 70s of block)
            fatiha_zone_end = min(bstart + 70.0, bend)
            windows = transcribe_block_windows(model_and_proc, wav_path, bstart, fatiha_zone_end, small_windows=True)
            # Large windows for the rest (just to confirm it's Quran)
            if fatiha_zone_end < bend - 5.0:
                rest_windows = transcribe_block_windows(model_and_proc, wav_path, fatiha_zone_end, bend, small_windows=False)
                windows.extend(rest_windows)
        else:
            windows = transcribe_block_windows(model_and_proc, wav_path, bstart, bend)

        for w in windows:
            print(f"[pipeline]   Window {w['abs_start']:.1f}-{w['abs_end']:.1f}s: {w['text'][:70]}")

        # Classify (may split block into fatiha + surah)
        parts = classify_block(windows, bstart, bend, is_first_recitation, remove_fatiha, fine_silences)

        for p in parts:
            all_segments.append(p)
            print(f"[pipeline]   → [{p['label']:>7}] {p['start']:.1f}-{p['end']:.1f}s")

        # After a recitation block, the next one starts a new rak'ah
        # (the structural silence between them was ruku/sujood)
        is_first_recitation = True

        # But if this was the first recitation (had Fatiha), the NEXT block
        # in the same rak'ah is the continuation (not a new rak'ah).
        # Actually, in taraweeh each rak'ah has ONE recitation block
        # (Fatiha + Surah), separated from the next rak'ah by ruku/sujood.
        # So after each big recitation block, next one IS a new rak'ah.

    # Step 5: Filter
    update_fn("filtering", 0.78)
    kept = [(s["start"], s["end"]) for s in all_segments if s["label"] == "quran"]
    removed = [s for s in all_segments if s["label"] != "quran"]

    print(f"\n[pipeline] KEPT {len(kept)} segments, REMOVED {len(removed)}")
    for s in all_segments:
        tag = "✓" if s["label"] == "quran" else "✗"
        print(f"[pipeline]   {tag} [{s['label']:>7}] {s['start']:.1f}-{s['end']:.1f}s")

    if not kept:
        raise RuntimeError(
            "Could not identify any Quran recitation segments. "
            "The audio may be too short or not contain clear recitation."
        )

    # Step 6: Stitch
    update_fn("stitching", 0.82)
    output_path = job_dir / "output.mp3"
    stitch_segments(wav_path, kept, output_path)
    update_fn("stitching", 0.95)

    kept_duration = sum(e - s for s, e in kept)
    removed_duration = sum(s["end"] - s["start"] for s in removed)

    wav_path.unlink(missing_ok=True)
    audio_path.unlink(missing_ok=True)

    return {
        "segmentsKept": len(kept),
        "segmentsRemoved": len(removed),
        "durationSecs": round(kept_duration, 1),
        "removedSecs": round(removed_duration, 1),
    }
