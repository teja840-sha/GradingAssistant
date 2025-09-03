
from __future__ import annotations
import os, argparse, json, pathlib, re, subprocess, time, math, wave, contextlib
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import yaml
import numpy as np
from PIL import Image, ImageDraw

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

# ---------- TTS + SRT ----------
def tts_to_wav(text: str, wav_path: str, rate: int = 170) -> None:
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

def wav_read(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    maxv = np.max(np.abs(audio)) or 1.0
    audio = audio / maxv
    return audio, fr

def compute_features(audio: np.ndarray, sr: int, hop_ms: int = 40, win_ms: int = 25):
    hop = int(sr * hop_ms / 1000.0)
    win = int(sr * win_ms / 1000.0)
    if win <= 0: win = int(sr * 0.025)
    feats = []
    for start in range(0, len(audio)-win, hop):
        seg = audio[start:start+win]
        if len(seg) < win: break
        rms = float(np.sqrt(np.mean(seg**2) + 1e-8))
        zc = float(((seg[:-1] * seg[1:]) < 0).sum() / (len(seg)-1))
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        mag = np.abs(spec) + 1e-8
        freqs = np.fft.rfftfreq(len(seg), d=1.0/sr)
        centroid = float((freqs * mag).sum() / mag.sum())
        feats.append((rms, zc, centroid))
    return feats, hop

def map_features_to_viseme(rms: float, zc: float, centroid: float) -> str:
    if rms < 0.02:
        return "M"
    if centroid > 3000 and zc > 0.12:
        return "E"
    if 1800 < centroid <= 3000:
        return "A"
    if 900 < centroid <= 1800:
        return "WU"
    return "O"

def smooth_sequence(seq: List[str]) -> List[str]:
    if not seq: return seq
    out = seq[:]
    for i in range(1, len(seq)-1):
        if seq[i-1] == seq[i+1] != seq[i]:
            out[i] = seq[i-1]
    return out

def draw_frame(base_img, viseme: str, mouth_box: tuple[float,float,float,float]):
    x, y, w, h = mouth_box
    W, H = base_img.size
    mx, my, mw, mh = int(x*W), int(y*H), int(w*W), int(h*H)
    frame = base_img.copy()
    dr = ImageDraw.Draw(frame, "RGBA")
    color = (180, 30, 30, 230)
    if viseme == "M":
        dr.rounded_rectangle([mx, my + mh//2 - mh//12, mx+mw, my + mh//2 + mh//12], radius=mh//8, fill=color)
    elif viseme == "A":
        dr.ellipse([mx+mw*0.25, my, mx+mw*0.75, my+mh], fill=color)
    elif viseme == "E":
        dr.ellipse([mx, my+mh*0.3, mx+mw, my+mh*0.7], fill=color)
    elif viseme == "O":
        r = min(mw, mh) * 0.55
        cx, cy = mx + mw//2, my + mh//2
        dr.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    elif viseme == "WU":
        r = min(mw, mh) * 0.35
        cx, cy = mx + mw//2, my + mh//2
        dr.ellipse([cx-r, cy-r, cx+r, cy+r], fill=color)
    else:
        dr.rounded_rectangle([mx, my + mh//2 - mh//16, mx+mw, my + mh//2 + mh//16], radius=mh//10, fill=color)
    return frame

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("FFmpeg not found on PATH. Please install ffmpeg and retry.")

def render_viseme_video_with_subtitles(avatar_path: str, wav_path: str, srt_path: str, out_mp4: str,
                                       fps: int = 25, mouth_box: tuple[float,float,float,float]=(0.38,0.62,0.24,0.14)):
    ensure_ffmpeg()
    base = Image.open(avatar_path).convert("RGB")
    audio, sr = wav_read(wav_path)
    feats, hop = compute_features(audio, sr)
    visemes = [map_features_to_viseme(r,z,c) for (r,z,c) in feats]
    visemes = smooth_sequence(visemes)
    dur_s = len(audio) / sr
    n_frames = int(math.ceil(fps * dur_s))
    frames_dir = pathlib.Path(out_mp4).with_suffix("").parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        t = i / fps
        idx = min(int(t * sr / hop), max(0, len(visemes)-1)) if hop > 0 and visemes else 0
        v = visemes[idx] if visemes else "REST"
        frm = draw_frame(base, v, mouth_box)
        frm.save(frames_dir / f"f_{i:05d}.png")

    img_glob = str(frames_dir / "f_%05d.png").replace("\\","/")
    srt = srt_path.replace("\\","/")
    cmd = [
        "ffmpeg","-y",
        "-r", str(fps), "-i", img_glob,
        "-i", wav_path,
        "-vf", f"subtitles='{srt}'",
        "-c:v","libx264","-pix_fmt","yuv420p",
        "-c:a","aac",
        "-shortest", out_mp4
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        cmd2 = [
            "ffmpeg","-y",
            "-r", str(fps), "-i", img_glob,
            "-i", wav_path,
            "-i", srt_path,
            "-c:v","libx264","-pix_fmt","yuv420p",
            "-c:a","aac",
            "-c:s","mov_text",
            "-shortest", out_mp4
        ]
        subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    load_dotenv(ROOT / "configs" / ".env", override=False)

    ap = argparse.ArgumentParser(description="App B: Avatar Viseme Renderer (CPU lipsync + SRT)")
    ap.add_argument("--text", type=str, default="This is a test narration for the viseme renderer.")
    ap.add_argument("--avatar", type=str, default=str(ROOT / "assets" / "ramanujan.jpg"))
    ap.add_argument("--outdir", type=str, default=str(OUTPUTS))
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--mouth_box", type=str, default="0.38,0.62,0.24,0.14", help="x,y,w,h fractions")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    conv_dir = pathlib.Path(args.outdir) / f"conv-{int(time.time())}"
    conv_dir.mkdir(parents=True, exist_ok=True)

    wav_path = str(conv_dir / "narration.wav")
    srt_path = str(conv_dir / "subtitles.srt")
    mp4_path = str(conv_dir / "video_viseme.mp4")

    # TTS + SRT
    tts_to_wav(args.text, wav_path)
    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        duration = wf.getnframes() / float(wf.getframerate() or 1)
    build_srt_from_text_and_duration(args.text, duration, srt_path)

    mouth_box = tuple(float(x.strip()) for x in args.mouth_box.split(","))  # type: ignore
    render_viseme_video_with_subtitles(args.avatar, wav_path, srt_path, mp4_path, fps=args.fps, mouth_box=mouth_box)  # type: ignore

    run_info = {
        "app": "app_avatar_viseme",
        "avatar": args.avatar,
        "outdir": str(conv_dir),
        "artifact": mp4_path,
        "duration_s": round(duration,2),
        "mouth_box": mouth_box,
    }
    print(json.dumps(run_info, indent=2))

if __name__ == "__main__":
    main()
