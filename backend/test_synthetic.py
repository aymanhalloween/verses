"""
Synthetic end-to-end test for the pipeline.

Builds a fake Taraweeh rak'ah with ffmpeg — sine tones simulate speech,
realistic silences mark structural boundaries. Monkeypatches `transcribe`
with canned Arabic text so we can exercise classify/stitch without needing
HuggingFace access.

Layout (absolute timestamps in seconds):

Layout reflects Taraweeh structure: Fatiha+Surah form one speech block
separated only by the ameen pause (~1s); ruku/sujood produce multi-second
structural gaps. The MERGE_GAP_SEC=1.8s in pipeline.py should join Fatiha
and Surah into one segment, then find the ameen pause inside.

Expected:
  - Rak'ah 1 (Fatiha 3–30 + ameen + Surah 31.2–90.2) becomes one block.
    Fatiha removed, surah (≈59s) kept.
  - Rak'ah 2 (Fatiha+Surah, ≈90s) kept after Fatiha trim (≈60s).
  - Total kept ≈ 120s.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import pipeline


def mksine(path: Path, dur: float, freq: int = 220):
    """Generate `dur` seconds of sine at `freq` Hz."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency={freq}:duration={dur}:sample_rate=16000",
        "-ac", "1", "-c:a", "pcm_s16le", str(path),
    ], capture_output=True, check=True)


def mksilence(path: Path, dur: float):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=16000:cl=mono",
        "-t", str(dur), "-c:a", "pcm_s16le", str(path),
    ], capture_output=True, check=True)


def build_synthetic_rakah(work: Path) -> Path:
    """Concat tones and silences into a fake Taraweeh."""
    parts: list[tuple[str, Path, float, int]] = [
        ("takbir",       work / "01.wav",  2.0, 180),
        ("silence",      work / "02.wav",  1.0, 0),
        ("fatiha",       work / "03.wav", 27.0, 240),
        ("ameen_pause",  work / "04.wav",  1.2, 0),
        ("surah",        work / "05.wav", 59.0, 300),
        ("pre_ruku",     work / "06.wav",  2.0, 0),
        ("ruku_silence", work / "07.wav",  8.0, 0),
        ("ruku_dhikr",   work / "08.wav", 10.0, 200),
        ("silence_a",    work / "09.wav",  2.0, 0),
        ("sujood",       work / "10.wav",  8.0, 200),
        ("silence_b",    work / "11.wav",  3.0, 0),
        ("rak2_fat",     work / "12a.wav", 25.0, 280),
        ("rak2_ameen",   work / "12b.wav",  1.0, 0),
        ("rak2_surah",   work / "12c.wav", 60.0, 260),
        ("closing_sil",  work / "13.wav",  1.5, 0),
        ("closing",      work / "14.wav",  3.5, 200),
    ]

    for name, p, dur, freq in parts:
        if freq == 0:
            mksilence(p, dur)
        else:
            mksine(p, dur, freq)

    listfile = work / "list.txt"
    listfile.write_text("".join(f"file '{p[1]}'\n" for p in parts))

    out = work / "fake_taraweeh.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-c:a", "pcm_s16le", str(out),
    ], capture_output=True, check=True)
    return out


# Canned transcripts. Keyed by (start, end) rounded to nearest second.
# For our synthetic rakah, the only calls we expect:
#   - find_fatiha_end check on 5s before boundary  → return Fatiha markers
#   - medium-segment classify calls                → label with dhikr markers
FAKE_TEXTS: dict[tuple[int, int], str] = {}


def fake_transcribe(wav, start, end):
    """Route slice timestamps to canned Arabic based on synthetic timeline."""
    # Fatiha check for rak'ah 1: slice ends near 31.2s (ameen boundary)
    if 26 <= end <= 32 and (end - start) <= 5.5:
        return "الحمد لله رب العالمين الرحمن الرحيم الضالين"
    # Fatiha check for rak'ah 2: slice ends near start+30 of rak2 block
    # (rak2 starts at 123.2 after merging, ameen at ~150ish)
    if 145 <= end <= 165 and (end - start) <= 5.5:
        return "الحمد لله الرحمن الرحيم الضالين"
    # Ruku/sujood dhikr check
    if 95 <= start <= 125:
        return "سبحان ربي العظيم الله اكبر"
    # Closing
    if start >= 155:
        return "السلام عليكم ورحمة الله"
    return ""


def main():
    work = Path("/tmp/quranclip_syn")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    print(">> Building synthetic rak'ah...")
    fake = build_synthetic_rakah(work)
    size_mb = fake.stat().st_size / 1024 / 1024
    print(f"   {fake} ({size_mb:.1f} MB)")

    job = work / "job"
    job.mkdir()
    inp = job / "input.audio"
    shutil.copy(fake, inp)

    def on_update(stage, pct):
        print(f"   [{stage:>13}] {pct*100:5.1f}%")

    print(">> Running pipeline (remove_fatiha=True)...")
    with patch.object(pipeline, "transcribe", side_effect=fake_transcribe):
        result = pipeline.process_audio(inp, job, remove_fatiha=True,
                                        update_fn=on_update)

    print(">> Result:")
    for k, v in result.items():
        print(f"   {k}: {v}")

    out = job / "output.mp3"
    assert out.exists(), "no output.mp3 produced"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True,
    )
    out_dur = float(probe.stdout.strip())
    print(f"   output.mp3 duration: {out_dur:.1f}s")

    # Expected: ~59s (surah of rak1) + ~60s (surah of rak2) ≈ 120s.
    assert 95 <= out_dur <= 145, f"output duration {out_dur:.1f}s out of expected range"
    assert result["segmentsRemoved"] >= 3, "should have dropped openings/dhikr/fatiha"
    assert result["segmentsKept"] == 2, f"should have 2 kept blocks, got {result['segmentsKept']}"
    print("PASS")


if __name__ == "__main__":
    main()
