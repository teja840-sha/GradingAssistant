
from __future__ import annotations
import os, argparse, json, pathlib, re, subprocess, time
from typing import Dict, Any
from dotenv import load_dotenv
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "configs" / "model_config.yaml"
OUTPUTS = ROOT / "outputs"

def _expand_env_vars_in_yaml(text: str) -> str:
    def repl(m):
        body = m.group(1)  # VAR or VAR:-default
        if ":-" in body:
            var, default = body.split(":-", 1)
            return os.getenv(var, default)
        return os.getenv(body, "")
    return re.sub(r"\$\{([^}]+)\}", repl, text)

def load_model_config(path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    rendered = _expand_env_vars_in_yaml(raw)
    return yaml.safe_load(rendered)

# ---------- Minimal TTS + SRT ----------
def tts_to_wav(text: str, wav_path: str, rate: int = 170) -> None:
    """Offline TTS using pyttsx3 (SAPI5 on Windows). Writes a WAV file."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.save_to_file(text, wav_path)
    engine.runAndWait()

def _format_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def build_srt_from_text_and_duration(text: str, duration_s: float, srt_path: str) -> None:
    import re as _re
    sentences = [s.strip() for s in _re.split(r'(?<=[\.\?\!])\s+', text.strip()) if s.strip()]
    if not sentences:
        sentences = [text.strip() or ""]
    wc = [max(1, len(_re.findall(r"\w+", s))) for s in sentences]
    total = sum(wc) or len(sentences)
    cues = []
    t = 0.0
    for i, s in enumerate(sentences):
        part = wc[i] / total
        seg = max(1.0, duration_s * part)
        start = t
        end = min(duration_s, start + seg)
        if i == len(sentences) - 1:
            end = duration_s
        cues.append((start, end, s))
        t = end
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (st, et, line) in enumerate(cues, 1):
            f.write(f"{idx}\n{_format_ts(st)} --> {_format_ts(et)}\n{line}\n\n")

def wav_duration_seconds(wav_path: str) -> float:
    import wave, contextlib
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate or 1)

# ---------- Static Image + Subtitles Renderer ----------
def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("FFmpeg not found on PATH. Please install ffmpeg and retry.")

def render_static_with_subtitles(image_path: str, wav_path: str, srt_path: str, out_mp4: str, fps: int = 25):
    """Produces an MP4 that shows a still image while playing audio, with subtitles burned in."""
    ensure_ffmpeg()
    img = image_path.replace("\\", "/")
    srt = srt_path.replace("\\", "/")
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(fps), "-i", img,
        "-i", wav_path,
        "-vf", f"subtitles='{srt}'",
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest", out_mp4,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # Fallback: embed soft subs
        cmd2 = [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(fps), "-i", img,
            "-i", wav_path,
            "-i", srt_path,
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-c:s", "mov_text",
            "-shortest", out_mp4,
        ]
        subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    load_dotenv(ROOT / "configs" / ".env", override=False)
    ap = argparse.ArgumentParser(description="App A: Grading Assistant (static image + SRT)")
    ap.add_argument("--avatar", type=str, default=str(ROOT / "assets" / "ramanujan.jpg"))
    ap.add_argument("--text", type=str, default="This is a short narrated summary for the Grading Assistant.")
    ap.add_argument("--outdir", type=str, default=str(OUTPUTS))
    args = ap.parse_args()

    cfg = load_model_config(CONFIG_FILE)
    os.makedirs(args.outdir, exist_ok=True)
    conv_dir = pathlib.Path(args.outdir) / f"conv-{int(time.time())}"
    conv_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(conv_dir / "narration.wav")
    srt_path = str(conv_dir / "subtitles.srt")
    mp4_path = str(conv_dir / "video_static.mp4")

    # 1) TTS
    tts_to_wav(args.text, wav_path)
    # 2) SRT
    dur = wav_duration_seconds(wav_path)
    build_srt_from_text_and_duration(args.text, dur, srt_path)
    # 3) Render still video + subs
    render_static_with_subtitles(args.avatar, wav_path, srt_path, mp4_path)

    run_info = {
        "app": "app_grading_assistant",
        "avatar": args.avatar,
        "outdir": str(conv_dir),
        "artifact": mp4_path,
    }
    print(json.dumps(run_info, indent=2))

if __name__ == "__main__":
    main()
