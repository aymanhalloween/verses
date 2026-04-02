"""Quick local test of the pipeline on a real audio file."""
import sys
import shutil
from pathlib import Path
from pipeline import process_audio

audio_src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/ayman/Desktop/International Pkwy 4.m4a")
if not audio_src.exists():
    print(f"File not found: {audio_src}")
    sys.exit(1)

# Set up a test job directory
job_dir = Path("/tmp/quranclip_test")
shutil.rmtree(job_dir, ignore_errors=True)
job_dir.mkdir(parents=True)

# Copy input file (pipeline will delete it after processing)
test_input = job_dir / "input.audio"
with open(audio_src, "rb") as src, open(test_input, "wb") as dst:
    dst.write(src.read())

def on_update(stage: str, progress: float):
    print(f"  [{stage:>13}] {progress*100:5.1f}%")

print(f"Processing: {audio_src.name} ({audio_src.stat().st_size / 1024 / 1024:.1f} MB)")
print("-" * 50)

try:
    result = process_audio(test_input, job_dir, remove_fatiha=True, update_fn=on_update)
    print("-" * 50)
    print(f"Done!")
    print(f"  Segments kept:    {result['segmentsKept']}")
    print(f"  Segments removed: {result['segmentsRemoved']}")
    print(f"  Clean duration:   {result['durationSecs']}s")
    print(f"  Removed duration: {result['removedSecs']}s")
    print(f"  Output: {job_dir / 'output.mp3'}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
