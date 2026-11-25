"""Join the scene clips and lay the narration (one recording per scene) and the music under them.

    python scripts/mux_video_audio.py <folder with n1.mp3 ... n8.mp3> <music.wav> <out.mp4>
Scene lengths and where each recording starts come from report/video_timing.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "processed" / "video" / "clips"
LEAD = 0.5                       # seconds of picture before the narration starts in each scene


def main(narr_dir, music, out):
    timing = json.loads((ROOT / "report" / "video_timing.json").read_text())
    lst = CLIPS / "list.txt"
    lst.write_text("".join(f"file 's{i}.mp4'\n" for i in range(1, 9)))
    silent = CLIPS / "joined_silent.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(silent)], check=True)
    inputs, chains, start = [], [], 0.0
    for i in range(1, 9):
        t = timing[str(i)]
        inputs += ["-i", str(Path(narr_dir) / f"n{i}.mp3")]
        d = int((start + LEAD) * 1000)
        # atrim keeps the spoken part; adelay places it in the scene (all=1: both channels)
        chains.append(f"[{i}:a]atrim={t['audio_start']}:{t['audio_end']},asetpts=PTS-STARTPTS,"
                      f"afade=t=in:d=0.05,afade=t=out:st={t['audio_end'] - t['audio_start'] - 0.15}:d=0.15,"
                      f"aresample=48000,adelay={d}:all=1[n{i}]")
        start += t["dur"]
    music_idx = 9
    graph = ";".join(chains) + ";" + "".join(f"[n{i}]" for i in range(1, 9)) + \
        f"amix=inputs=8:normalize=0,apad=whole_dur={start:.3f},acompressor=threshold=0.1:ratio=3:attack=5:release=120,volume=1.4,asplit=2[voice][key];" + \
        f"[{music_idx}:a]volume=0.3[mus];" + \
        "[mus][key]sidechaincompress=threshold=0.02:ratio=5:attack=80:release=700[duck];" + \
        "[voice][duck]amix=inputs=2:normalize=0,alimiter=limit=0.95[a]"
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(silent)] + inputs + ["-i", music,
           "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", out]
    subprocess.run(cmd, check=True)
    print("wrote", out, f"({start:.1f} s)")


if __name__ == "__main__":
    main(*sys.argv[1:4])
