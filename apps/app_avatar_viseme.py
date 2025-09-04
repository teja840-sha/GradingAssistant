# apps/app_avatar_viseme.py - renders talking head videos with subtitles
from __future__ import annotations
import os, json, re, math, asyncio, time, sys, subprocess
from typing import List, TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, END

# ───────────── TOGGLE (no env vars) ─────────────
MOCK_MODE = False  # ← set True to run fully offline with mock agents

# ───────────── PATHS ─────────────
PROMPTS_DIR = "prompts"
ASSETS_DIR = "assets"
OUTPUT_DIR = "outputs"
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subject → avatar image (fallbacks if missing)
PERSONA_IMAGES = {
    "Mathematics": "mathematics.jpg",
    "Science": "science.png",
    "English Language Arts": "english.png",
    "Social Studies": "social_studies.png",
    "Computer Science": "computer_science.png",
}
def _pick_default_avatar() -> str:
    for fn in os.listdir(ASSETS_DIR):
        if fn.lower().endswith((".jpg",".jpeg",".png")):
            return os.path.join(ASSETS_DIR, fn)
    return os.path.join(ASSETS_DIR, "science.png")
DEFAULT_AVATAR = _pick_default_avatar()

# ───────────── Chainlit (optional UI) ─────────────
try:
    import chainlit as cl
except Exception:
    cl = None

# ───────────── LLM client ─────────────
class LLM:
    def __init__(self):
        if MOCK_MODE:
            self.llm = None
            return
        from langchain_openai import AzureChatOpenAI
        import yaml

        with open("model_config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["config"]
        self.llm = AzureChatOpenAI(
            azure_endpoint=cfg["azure_endpoint"],
            api_key=cfg["api_key"],
            api_version=cfg["api_version"],
            azure_deployment=cfg["azure_deployment"],
            model=cfg["model"],
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            seed=int(cfg.get("seed", 1)),
            streaming=False,
        )

    async def call_json(self, system: str, user: str, tag: str = "") -> dict:
        if MOCK_MODE:
            return self._mock(tag, user=user)
        msgs = [{"role":"system","content":system},{"role":"user","content":user}]
        resp = await self.llm.ainvoke(msgs)
        txt = (getattr(resp, "content", "") or "").strip()
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
            if txt.lower().startswith("json"):
                txt = txt[4:].strip()
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{[\s\S]*\}$", txt)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {"raw": txt}

    # ── Mock Agents (offline) ──
    def _mock(self, tag: str, user: str = "") -> dict:
        if tag == "extract_qas":
            return {
                "questions": [
                    {"id": f"Q{i}", "topic": t, "text": f"[{t}] problem text...", "max_marks": m}
                    for i, (t, m) in enumerate([
                        ("System of linear equations", 10),
                        ("Quadratic equations", 8),
                        ("Heron's formula, triangle area", 10),
                        ("Trigonometry identities", 8),
                        ("Arithmetic progression", 8),
                        ("Probability basics", 10),
                        ("Circle equation", 12),
                        ("Differentiation rules", 12),
                        ("Euclid's algorithm", 12),
                        ("Mean & Variance", 10),
                    ], start=1)
                ],
                "answers": [{"id": f"Q{i}", "text": f"Student answer for Q{i}..."} for i in range(1, 11)]
            }
        if tag == "strict_grade":
            payload = json.loads(user) if user else {}
            qas = payload.get("qas", [])
            marks_map = payload.get("marks_map", {})
            out = []
            for qa in qas:
                qid = qa["id"]
                out_of = int(marks_map.get(qid, 5))
                awarded = max(0, out_of - (2 if qid in {"Q3", "Q6"} else 0))
                out.append({
                    "id": qid, "topic": qa.get("topic","Topic"),
                    "awarded": awarded, "out_of": out_of,
                    "rationale": "Checked setup, steps and final answer.",
                    "fix": "Show intermediate steps clearly.",
                    "correctness": "correct" if awarded == out_of else "partially-correct"
                })
            return {"graded": out, "totals": {"awarded": sum(r["awarded"] for r in out), "out_of": sum(r["out_of"] for r in out)}}
        if tag == "plan":
            payload = json.loads(user) if user else {}
            gaps = payload.get("gaps", [])
            plans = []
            for g in gaps[:3]:
                topic = g["topic"]
                plans.append({
                    "topic": topic,
                    "goals": ["Concepts", "Worked examples", "Practice", "Self-check"],
                    "schedule": [
                        {"day": 1, "lesson": "Concept overview", "practice": 3},
                        {"day": 2, "lesson": "Worked examples", "practice": 4},
                        {"day": 3, "lesson": "Mixed practice", "practice": 5},
                        {"day": 4, "lesson": "Common errors & review", "practice": 4},
                        {"day": 5, "lesson": "Assessment + reflection", "practice": 5},
                    ],
                    "resources": [
                        {"title": "Khan Academy", "url": "https://www.khanacademy.org/"},
                        {"title": "Paul's Notes", "url": "https://tutorial.math.lamar.edu/"},
                    ],
                    "content_snippets": [
                        "Key definitions & formulae.",
                        "Two worked problems with commentary.",
                        "Mini-checklist for verifying answers."
                    ]
                })
            return {"plans": plans}
        if tag == "quiz":
            return {"questions": [{"q": f"Question {i}: Solve/Explain ...", "topic": "mixed"} for i in range(1, 21)]}
        return {}

llm = LLM()

# ───────────── Prompt loader ─────────────
def load_prompt_text(name: str) -> str:
    import yaml
    p = os.path.join(PROMPTS_DIR, f"{name}.yaml")
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            for k in ("system", "instructions", "prompt"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v
        with open(p, "r", encoding="utf-8") as f2:
            return f2.read()
    except Exception:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

# Fallbacks (used if yaml missing)
FALLBACK_EXTRACTOR = (
    "You are an EdTech extractor. From QUESTION_PDF_TEXT and ANSWER_PDF_TEXT, "
    "emit aligned Q/A with inferred max marks. JSON only:\n"
    "{ \"questions\":[{\"id\":\"Q1\",\"topic\":\"...\",\"text\":\"...\",\"max_marks\":10},...],"
    "  \"answers\":[{\"id\":\"Q1\",\"text\":\"...\"},...] }"
)
FALLBACK_GRADER = (
    "You are a strict grader. Grade each QA using a 5-dimension rubric "
    "(setup, steps, correctness, verification, reasoning). Penalize missing steps. JSON only:\n"
    "{\"graded\":[{\"id\":\"Q1\",\"topic\":\"...\",\"awarded\":4,\"out_of\":5,"
    "\"rationale\":\"short\",\"fix\":\"short\",\"correctness\":\"correct|partially-correct|incorrect\"}],"
    "\"totals\":{\"awarded\":...,\"out_of\":...}}"
)
FALLBACK_PLANNER = (
    "Create a weekly plan per weak topic with goals, 5-day schedule, and at least 2 credible references. "
    "Also include brief 'content_snippets' (2-3 lines) learners can read now. JSON only:\n"
    "{\"plans\":[{\"topic\":\"...\",\"goals\":[...],"
    "\"schedule\":[{\"day\":1,\"lesson\":\"...\",\"practice\":N},...],"
    "\"resources\":[{\"title\":\"..\",\"url\":\"..\"},...],"
    "\"content_snippets\":[\"...\",\"...\"]}]}"
)
FALLBACK_QUIZ = (
    "Generate 20 varied, challenging questions for the given subject+grade. JSON only:\n"
    "{\"questions\":[{\"q\":\"...\",\"topic\":\"...\"},...]}"
)

# ───────────── Helpers ─────────────
def format_quiz_minimal(questions: list[dict]) -> str:
    """Render only '1. Question' lines."""
    return "\n".join(f"{i}. {q.get('text') or q.get('q') or ''}".strip()
                     for i, q in enumerate(questions, 1))

def read_pdf_text(path: str) -> str:
    import PyPDF2
    text = ""
    with open(path, "rb") as f:
        r = PyPDF2.PdfReader(f)
        for p in r.pages:
            text += p.extract_text() or ""
    return text

def as_int(x, default=0):
    try: return int(x)
    except Exception:
        try: return int(float(x))
        except Exception: return default

def percent(x):
    try: v = float(x); return max(0, min(100, int(round(v))))
    except Exception: return 0

def _as_list(x):
    if x is None: return []
    return x if isinstance(x, list) else [x]

def pick_avatar_for_subject(subject: str) -> str:
    fname = PERSONA_IMAGES.get(subject)
    if fname:
        candidate = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(candidate):
            return candidate
    for fn in os.listdir(ASSETS_DIR):
        if fn.lower().endswith((".jpg",".jpeg",".png")):
            return os.path.join(ASSETS_DIR, fn)
    return DEFAULT_AVATAR

def normalize_reply(res) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        for k in ("content", "output", "text", "message", "value", "reply"):
            v = res.get(k)
            if isinstance(v, str):
                return v.strip()
        for v in res.values():
            if isinstance(v, str):
                return v.strip()
    for k in ("content", "output", "text", "message", "value", "reply"):
        v = getattr(res, k, None)
        if isinstance(v, str):
            return v.strip()
    return str(res).strip()

# ───────────── Video / Audio ─────────────
def _ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("FFmpeg not found in PATH. Please install FFmpeg.")

def _tts_blocking_to_wav(text: str, wav_path: str) -> None:
    import pyttsx3
    engine = pyttsx3.init(driverName="sapi5" if os.name == "nt" else None)
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)
    engine.save_to_file(text, wav_path)
    engine.runAndWait()

def _format_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _make_srt_from_text_and_audio(text: str, audio_path: str, srt_path: str) -> None:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    clip = AudioFileClip(audio_path)
    duration = float(getattr(clip, "duration", 0.0) or 5.0)
    clip.close()
    sents = [s.strip() for s in re.split(r'(?<=[\.\?\!])\s+', (text or "").strip()) if s.strip()] or [text.strip() or ""]
    wc = [max(1, len(re.findall(r"\w+", s))) for s in sents]
    total_w = sum(wc) or len(sents)
    cues, t0 = [], 0.0
    for i, s in enumerate(sents):
        part = wc[i] / total_w
        seg = max(1.0, duration * part)
        st, et = t0, min(duration, t0+seg)
        if i == len(sents)-1: et = duration
        cues.append((st, et, s))
        t0 = et
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, (st, et, line) in enumerate(cues, 1):
            f.write(f"{idx}\n{_format_ts(st)} --> {_format_ts(et)}\n{line}\n\n")

def _burn_subs_ffmpeg(in_mp4: str, srt_path: str, out_mp4: str) -> bool:
    srt_arg = srt_path.replace("\\", "/")
    cmd = ["ffmpeg","-y","-i",in_mp4,"-vf",f"subtitles='{srt_arg}'","-c:a","copy",out_mp4]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _embed_soft_subs_ffmpeg(in_mp4: str, srt_path: str, out_mp4: str) -> bool:
    cmd = ["ffmpeg","-y","-i",in_mp4,"-i",srt_path,"-c","copy","-c:s","mov_text",out_mp4]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def _still_image_video(face_img: str, audio_path: str, out_mp4: str) -> None:
    subprocess.run([
        "ffmpeg","-y",
        "-loop","1","-i",face_img,
        "-i",audio_path,
        "-shortest",
        "-c:v","libx264","-tune","stillimage","-pix_fmt","yuv420p","-r","25",
        "-c:a","aac",
        out_mp4
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _ffprobe_size(img_path: str) -> tuple[int, int]:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", img_path
    ], stderr=subprocess.STDOUT).decode("utf-8").strip()
    w, h = out.split("x")
    return int(w), int(h)


def _lipsync_still_image_video(face_img: str, audio_path: str, out_mp4: str, debug_box: bool = False) -> None:
    """
    Create a still-image video but 'animate' lips by cropping a mouth region
    and vertically scaling it over time using a sinusoid. Base image remains static.
    No third-party Python libs; uses ffmpeg only.
    """
    # 1) Read image size safely
    W, H = _ffprobe_size(face_img)

    # 2) Heuristic mouth rectangle (lower-center of face)
    #    Tune these 4 numbers to match your avatar layout if needed.
    mw = max(20, int(W * 0.30))           # mouth width
    mh = max(10, int(H * 0.12))           # mouth height
    mx = max(0, (W - mw) // 2)            # x centered
    my = max(0, int(H * 0.62) - mh // 2)  # y ~ 62% from top

    # 3) Motion model m(t) = |sin(2π f t)|  in [0,1]
    freq = 3.5
    base = 0.75     # baseline vertical scale (1.0 = unchanged height)
    amp  = 0.45     # extra opening amplitude

    # 4) Filter graph:
    #    - Make a clean, static base from the still
    #    - Derive a mouth crop from the same frame
    #    - Scale the mouth vertically by base + amp*|sin(2π f t)|
    #    - Overlay the scaled mouth back, re-centering vertically
    # NOTE: crop **ORDER** is crop=w:h:x:y  (corrected!)
    mouth_crop = f"crop={mw}:{mh}:{mx}:{my}"
    mouth_scale = f"scale=w=iw:h=ih*({base}+{amp}*abs(sin(2*PI*{freq}*t))):eval=frame"

    # Keep base static RGBA for precise overlay math
    base_chain = "fps=25,format=rgba"
    if debug_box:
        # draw the intended mouth rectangle on base for sanity check
        base_chain += f",drawbox=x={mx}:y={my}:w={mw}:h={mh}:color=lime@0.5:t=2"

    # Re-center vertically: move up by half the added height
    # overlay_h is the current scaled-mouth height
    overlay_y = f"{my}-(overlay_h-{mh})/2"

    filter_complex = (
        f"[0:v]{base_chain},split[base][mouthsrc];"
        f"[mouthsrc]{mouth_crop},{mouth_scale}[mouth];"
        f"[base][mouth]overlay=x={mx}:y={overlay_y}:eval=frame,format=yuv420p[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", face_img,  # video stream from still
        "-i", audio_path,              # audio
        "-shortest",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "1:a:0",
        "-r", "25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        out_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def make_video_with_subs(text: str, face_img: str) -> str:
    _ensure_ffmpeg()
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(OUTPUT_DIR, f"plan_{ts}")
    wav_path = f"{stem}.wav"
    anim_path = f"{stem}_still.mp4"
    srt_path = f"{stem}.srt"
    final_path = f"{stem}_subtitled.mp4"

    if cl: await cl.Message(content="🎙️ Synthesizing narration...").send()
    await asyncio.to_thread(_tts_blocking_to_wav, text, wav_path)

    if cl: await cl.Message(content="🖼️ Rendering video from persona image...").send()
    try:
        await asyncio.to_thread(_lipsync_still_image_video, face_img, wav_path, anim_path, True)
    except Exception:
        await asyncio.to_thread(_still_image_video, face_img, wav_path, anim_path)

    if cl: await cl.Message(content="📝 Building subtitles...").send()
    _make_srt_from_text_and_audio(text, wav_path, srt_path)

    if cl: await cl.Message(content="🔥 Burning subtitles into the video...").send()
    burned = await asyncio.to_thread(_burn_subs_ffmpeg, anim_path, srt_path, final_path)
    if not burned:
        embedded = await asyncio.to_thread(_embed_soft_subs_ffmpeg, anim_path, srt_path, final_path)
        if not embedded:
            final_path = anim_path
    return final_path

# ───────────── Grading / Gaps ─────────────
def summarize_gaps_weighted(graded_rows: List[dict]) -> List[dict]:
    agg: dict = {}
    for r in graded_rows:
        t = r.get("topic", "Unknown")
        agg.setdefault(t, {"aw": 0.0, "max": 0.0})
        agg[t]["aw"] += float(r.get("awarded", 0))
        agg[t]["max"] += float(r.get("out_of", 0))
    gaps = []
    for topic, v in agg.items():
        mastery = (v["aw"] / v["max"]) if v["max"] > 0 else 0.0
        gaps.append({"topic": topic, "mastery": round(mastery, 2)})
    gaps.sort(key=lambda x: x["mastery"])
    return gaps

# ───────────── Quiz helpers ─────────────
def subject_to_key(subject: str) -> str:
    return (subject or "").strip().lower().replace(" ", "_")

def grade_to_key(standard: str) -> str:
    # "Grade 10" -> "grade_10"
    s = (standard or "").strip().lower()
    s = s.replace(" ", "_")
    if not s.startswith("grade_"):
        s = f"grade_{re.sub(r'\\D', '', s) or '9'}"
    return s

def parse_quiz_payload(state: dict) -> dict:
    """Build payload that matches your YAML spec."""
    payload = {
        "subject_key": subject_to_key(state.get("subject", "")),
        "grade_key": grade_to_key(state.get("standard", "")),
        "target_items": 20,
        "difficulty_policy": {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    }
    # optional: if you store a focus topic in state
    if state.get("topic"):
        payload["topic"] = state["topic"]
    # optional: topics catalog if you ever set it
    if state.get("topics_catalog"):
        payload["topics_catalog"] = state["topics_catalog"]
    return payload

def _coerce_questions(qz: dict, n: int = 20) -> list[dict]:
    """Extract questions list and normalize the keys we rely on."""
    qs = []
    for item in (qz or {}).get("questions", []):
        if not isinstance(item, dict):
            continue
        # prefer 'text', fallback to 'q'
        text = item.get("text") or item.get("q") or ""
        if text:
            qs.append({"id": item.get("id"), "text": text})
    # trim/pad
    qs = qs[:n]
    while len(qs) < n:
        i = len(qs) + 1
        qs.append({"id": f"Q{i}", "text": f"Question {i}: (no text)"})
    return qs

def format_quiz_minimal(questions: list[dict]) -> str:
    """Render only '1. Question' lines."""
    return "\n".join(f"{i}. {q.get('text','')}".strip()
                     for i, q in enumerate(questions, 1))

def extract_quiz_meta(qz: dict) -> tuple[list[dict], list[str]]:
    """Return (answer_key, video_points)."""
    ak = []
    for it in (qz or {}).get("answer_key", []):
        if isinstance(it, dict) and ("id" in it or "expected" in it):
            ak.append({"id": it.get("id"), "expected": it.get("expected", "")})
    vpts = [str(x) for x in (qz or {}).get("video_points", []) if str(x).strip()]
    return ak, vpts


def _parse_numbered_answers(s: str, n: int) -> dict:
    """
    Accept lines like:
      1) answer
      1. answer
      1 - answer
      1: answer
    """
    out: dict = {}
    for line in (s or "").splitlines():
        m = re.match(r"\s*(\d{1,2})\s*[\)\.\-:]\s*(.+?)\s*$", line)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n:
                out[f"Q{idx}"] = m.group(2)
    return out

# ───────────── LangGraph orchestration ─────────────
class AppState(TypedDict, total=False):
    subject: str
    standard: str
    mode: str                 # "upload" or "quiz"
    q_text: str               # question paper text
    a_text: str               # answers text
    qa: dict
    graded: dict
    gaps: list[dict]
    plan: dict
    plan_summary: str
    narration_text: str
    avatar_image: str
    quiz: dict

async def node_extract(state: AppState) -> AppState:
    if cl: await cl.Message(content="🔎 **Extractor Agent**: parsing Q&A PDFs...").send()
    sys_prompt = load_prompt_text("extractor") or FALLBACK_EXTRACTOR
    user = f"QUESTION_PDF_TEXT:\n{(state.get('q_text') or '')[:15000]}\n\nANSWER_PDF_TEXT:\n{(state.get('a_text') or '')[:15000]}"
    qa = await llm.call_json(sys_prompt, user, tag="extract_qas")
    state["qa"] = qa
    return state

async def node_grade(state: AppState) -> AppState:
    if cl: await cl.Message(content="🧮 **Grader Agent**: applying rubric...").send()
    sys_prompt = load_prompt_text("grader") or FALLBACK_GRADER
    questions = state.get("qa", {}).get("questions", [])
    answers = state.get("qa", {}).get("answers", [])
    ans_map = {a.get("id"): a.get("text","") for a in answers if isinstance(a, dict)}
    qas, marks_map = [], {}
    for q in questions:
        qid = q.get("id")
        qas.append({"id": qid, "topic": q.get("topic","Topic"),
                    "question": q.get("text",""), "answer": ans_map.get(qid,"")})
        marks_map[qid] = as_int(q.get("max_marks", 5), 5)
    payload = json.dumps({"qas": qas, "marks_map": marks_map}, ensure_ascii=False)
    graded = await llm.call_json(sys_prompt, payload, tag="strict_grade")
    state["graded"] = graded
    return state

async def node_gaps(state: AppState) -> AppState:
    if cl: await cl.Message(content="📉 **Gap Analyzer Agent**: computing mastery by topic...").send()
    graded_rows = state.get("graded", {}).get("graded", [])
    state["gaps"] = summarize_gaps_weighted(graded_rows)
    return state

async def node_plan(state: AppState) -> AppState:
    if cl: await cl.Message(content="🗺️ **Planner Agent**: preparing weekly course plan with references & starter content...").send()
    sys_prompt = load_prompt_text("planner") or FALLBACK_PLANNER
    payload = json.dumps({"gaps": state.get("gaps", []),
                          "subject": state.get("subject",""),
                          "standard": state.get("standard","")}, ensure_ascii=False)
    plan = await llm.call_json(sys_prompt, payload, tag="plan")
    state["plan"] = plan
    # Short narration from plan
    lines = []
    for p in _as_list(plan.get("plans", [])):
        topic = p.get("topic","Topic")
        goals = p.get("goals") or ["concepts","examples","practice"]
        lines.append(f"For {topic}, goals include {', '.join(goals)}.")
    state["plan_summary"] = " ".join(lines) or "This plan covers your weak areas with focused practice."
    state["narration_text"] = state["plan_summary"]
    return state

async def node_quiz(state: AppState) -> AppState:
    if cl: await cl.Message(content="❓ **Quiz Generator Agent**: composing 20 questions...").send()
    sys_prompt = load_prompt_text("quiz") or FALLBACK_QUIZ
    user = json.dumps(parse_quiz_payload(state), ensure_ascii=False)
    qz = await llm.call_json(sys_prompt, user, tag="quiz")
    state["quiz"] = qz
    return state


def build_graph():
    g = StateGraph(AppState)
    # nodes
    g.add_node("router", lambda s: s)  # no-op
    g.add_node("extract", node_extract)
    g.add_node("grade", node_grade)
    g.add_node("gaps", node_gaps)
    g.add_node("plan", node_plan)
    g.add_node("quiz", node_quiz)

    def route_decision(state: AppState) -> str:
        return "quiz" if state.get("mode") == "quiz" else "upload"

    g.add_conditional_edges(
        "router",
        route_decision,
        {"quiz": "quiz", "upload": "extract"},
    )

    g.add_edge("extract", "grade")
    g.add_edge("grade", "gaps")
    g.add_edge("gaps", "plan")
    g.add_edge("plan", END)
    g.add_edge("quiz", END)

    g.set_entry_point("router")
    return g.compile()

GRAPH = build_graph()

# ───────────── Presentation helpers ─────────────
async def present_grades_table(graded: List[dict], totals: dict):
    if not cl: return
    lines = ["| Q | Topic | Score |", "|---|---|---:|"]
    for row in graded:
        lines.append(f"| {row.get('id','-')} | {row.get('topic','-')} | {row.get('awarded',0)}/{row.get('out_of',0)} |")
    lines.append(f"| **Total** |  | **{totals.get('awarded',0)}/{totals.get('out_of',0)}** |")
    await cl.Message(content="\n".join(lines)).send()

async def present_feedback_points(graded: List[dict]):
    if not cl: return
    bullets = []
    for i, r in enumerate(graded, 1):
        bullets.append(
            f"**{r.get('id', f'Q{i}')}: {r.get('topic','')}** — {r.get('rationale','')}\n"
            f"*Fix:* {r.get('fix','')}\n"
            f"*Score:* {r.get('awarded',0)}/{r.get('out_of',0)}  _({r.get('correctness','graded')})_"
        )
    await cl.Message(content="\n\n".join(bullets)).send()

async def present_lagging(gaps: List[dict]):
    if not cl: return
    if not gaps:
        await cl.Message(content="No detectable gaps. Great job!").send()
        return
    lines = ["### Where you’re **lagging / need review**"]
    for g in gaps:
        lvl = "weak" if g["mastery"] < 0.7 else "needs review" if g["mastery"] < 0.9 else "strong"
        lines.append(f"- **{g['topic']}** — mastery **{percent(g['mastery']*100)}%** ({lvl})")
    await cl.Message(content="\n".join(lines)).send()

async def present_plan(plan: dict):
    if not cl: return
    plans = _as_list(plan.get("plans", []))
    for p in plans:
        topic = p.get("topic","Topic")
        goals = ", ".join(p.get("goals", [])) or "(customized targets)"
        sched_lines = [f"- **Day {i.get('day',idx+1)}**: {i.get('lesson','')}"
                       f" (practice ×{i.get('practice',0)})"
                       for idx, i in enumerate(_as_list(p.get("schedule", [])))]
        res_lines = [f"- [{r.get('title','link')}]({r.get('url','')})" if r.get("url") else f"- {r.get('title','link')}"
                     for r in _as_list(p.get("resources", []))]
        snippets = "\n".join([f"- {s}" for s in _as_list(p.get("content_snippets", []))])
        await cl.Message(content=(
            f"## Personalized plan: {topic}\n"
            f"**Goals:** {goals}\n\n"
            f"**Schedule:**\n{'\n'.join(sched_lines) if sched_lines else '- (none)'}\n\n"
            f"**References:**\n{'\n'.join(res_lines) if res_lines else '- (none)'}\n\n"
            f"**Starter content:**\n{snippets if snippets else '- (none)'}\n"
        )).send()

# ───────────── Restart helper ─────────────
async def restart_ui():
    # Reset keys we use
    for k in ("subject","standard","mode","avatar_image","quiz_questions","awaiting_quiz_answers"):
        try:
            del cl.user_session[k]
        except Exception:
            pass
    await start()  # call the same function we registered with @cl.on_chat_start

# ───────────── Chainlit UI flow ─────────────
if cl:
    @cl.on_chat_start
    async def start():
        subjects = ["Mathematics","English Language Arts","Science","Social Studies","Computer Science"]
        cl.user_session.set("subjects", subjects)
        cl.user_session.set("standards", {
            "Mathematics": ["Grade 6","Grade 7","Grade 8","Grade 9","Grade 10"],
            "English Language Arts": ["Grade 9","Grade 10"],
            "Science": ["Grade 9","Grade 10"],
            "Social Studies": ["Grade 9","Grade 10"],
            "Computer Science": ["Grade 9","Grade 10"],
        })
        cl.user_session.set("mode", None)

        actions = [cl.Action(name="choose_subject", label=s, payload={"subject": s}) for s in subjects]
        welcome = (
            "**Welcome to Grading Assistant**\n"
            "I can grade these subjects:\n- " + "\n- ".join(subjects) +
            "\n\nPick a subject to begin."
        )
        await cl.Message(content=welcome, actions=actions).send()

    @cl.action_callback("choose_subject")
    async def on_subject(action: cl.Action):
        subject = action.payload["subject"]
        cl.user_session.set("subject", subject)
        avatar = pick_avatar_for_subject(subject)
        cl.user_session.set("avatar_image", avatar)

        stds = cl.user_session.get("standards", {}).get(subject, ["Grade 9"])
        std_actions = [cl.Action(name="choose_standard", label=s, payload={"standard": s}) for s in stds]
        await cl.Message(content=f"**{subject}** selected. Choose your standard:", actions=std_actions).send()

    @cl.action_callback("choose_standard")
    async def on_standard(action: cl.Action):
        standard = action.payload["standard"]
        cl.user_session.set("standard", standard)
        subject = cl.user_session.get("subject")

        await cl.Message(content=(
            f"Subject **{subject}**, Standard **{standard}**.\n"
            "Would you like to **Upload PDFs** (question + answer) or **Generate Tailored Quiz**?"
        )).send()

        await cl.Message(
            content="",
            actions=[
                cl.Action(name="choose_mode", label="Upload Q&A PDFs", payload={"mode": "upload"}),
                cl.Action(name="choose_mode", label="Generate Tailored Quiz", payload={"mode": "quiz"}),
            ],
        ).send()

    @cl.action_callback("choose_mode")
    async def on_mode(action: cl.Action):
        mode = action.payload["mode"]
        cl.user_session.set("mode", mode)

        subject = cl.user_session.get("subject") or ""
        standard = cl.user_session.get("standard") or ""
        avatar = cl.user_session.get("avatar_image") or DEFAULT_AVATAR

        if mode == "upload":
            qf = await cl.AskFileMessage(
                content="Upload **Question PDF**.",
                accept=["application/pdf"], max_size_mb=25
            ).send()
            af = await cl.AskFileMessage(
                content="Upload **Answer PDF** (student answers).",
                accept=["application/pdf"], max_size_mb=25
            ).send()
            if not qf or not af:
                await cl.Message(content="I need both PDFs to proceed.").send()
                return

            await cl.Message(content="📥 Reading PDFs...").send()
            qtext = read_pdf_text(qf[0].path)
            atext = read_pdf_text(af[0].path)

            init: AppState = {"subject": subject, "standard": standard, "mode":"upload",
                              "q_text": qtext, "a_text": atext, "avatar_image": avatar}

            # Full pipeline (Extractor → Grader → Gaps → Plan)
            result = await GRAPH.ainvoke(init)

            graded_rows = result.get("graded", {}).get("graded", [])
            totals = result.get("graded", {}).get("totals", {"awarded":0,"out_of":0})
            await present_grades_table(graded_rows, totals)
            await present_feedback_points(graded_rows)
            await present_lagging(result.get("gaps", []))

            # Plan (with references + starter content)
            await present_plan(result.get("plan", {}))

            # Video?
            ask_vid = cl.AskUserMessage(
                content="Create a short **explainer video with subtitles** from the plan? (yes/no)",
                timeout=300
            )
            vid_resp = await ask_vid.send()
            if normalize_reply(vid_resp).lower() in {"y","yes"}:
                path = await make_video_with_subs(result.get("plan_summary","Your plan summary."), avatar)
                try:
                    vid_el = cl.Video(name=os.path.basename(path), path=path, display="inline")
                except Exception:
                    vid_el = cl.File(name=os.path.basename(path), path=path, display="inline", mime="video/mp4")
                await cl.Message(content="Your narrated video is ready:", elements=[vid_el]).send()

            # Restart?
            ask_restart = cl.AskUserMessage(
                content="Want to **start over** and try a different mode/subject? (yes/no)",
                timeout=300
            )
            res = await ask_restart.send()
            if normalize_reply(res).lower() in {"y","yes"}:
                await restart_ui()
            else:
                await cl.Message(content="Okay—type **/reset** anytime to begin again.").send()

        else:  # quiz
            init: AppState = {
                "subject": subject,
                "standard": standard,
                "mode": "quiz",
                "avatar_image": avatar,
            }
            result = await GRAPH.ainvoke(init)

            qz = result.get("quiz", {}) or {}
            questions = _coerce_questions(qz, 20)
            answer_key, video_points = extract_quiz_meta(qz)

            # store for grading later
            cl.user_session.set("quiz_questions", questions)
            cl.user_session.set("quiz_answer_key", answer_key)
            cl.user_session.set("quiz_video_points", video_points)
            cl.user_session.set("awaiting_quiz_answers", True)

            # minimal rendering: only "1. Question"
            list_md = format_quiz_minimal(questions)
            await cl.Message(content=f"### Your quiz (20 Qs)\n{list_md}").send()

            await cl.Message(content=(
                "Reply with your **numbered answers in one message**, e.g.\n"
                "`1) ... 2) ... 3) ...` (any of `1)`, `1.`, `1-`, or `1:` is fine)."
            )).send()


    @cl.on_message
    async def on_free_message(message: cl.Message):
        # Only capture when we're waiting for quiz answers
        if not cl.user_session.get("awaiting_quiz_answers"):
            return

        questions: List[dict] = cl.user_session.get("quiz_questions") or []
        if not questions:
            await cl.Message(content="No quiz is active. Type **/reset** to start.").send()
            return

        ans_map = _parse_numbered_answers(message.content or "", len(questions))
        if not ans_map:
            await cl.Message(content="I couldn't detect numbered answers. Please send lines like `1) your answer`.").send()
            return

        # Build QAs for grading
        if cl: await cl.Message(content="🧮 **Quiz Grader Agent**: evaluating your answers...").send()
        qas, marks_map = [], {}
        for i, q in enumerate(questions, 1):
            qid = f"Q{i}"
            qas.append({
                "id": qid,
                "topic": q.get("topic","mixed"),
                "question": q.get("text",""),
                "answer": ans_map.get(qid, "")
            })
            marks_map[qid] = 5  # uniform 5 marks per quiz item

        sys_prompt = load_prompt_text("grader") or FALLBACK_GRADER
        payload = json.dumps({"qas": qas, "marks_map": marks_map}, ensure_ascii=False)
        graded = await llm.call_json(sys_prompt, payload, tag="strict_grade")

        rows = graded.get("graded", [])
        totals = graded.get("totals", {"awarded":0,"out_of":0})
        await present_grades_table(rows, totals)
        await present_feedback_points(rows)

        # Gaps + Plan
        gaps = summarize_gaps_weighted(rows)
        await present_lagging(gaps)

        if cl: await cl.Message(content="🗺️ **Planner Agent**: preparing weekly plan from your quiz performance...").send()
        sys_prompt_pl = load_prompt_text("planner") or FALLBACK_PLANNER
        plan = await llm.call_json(sys_prompt_pl, json.dumps({
            "gaps": gaps,
            "subject": cl.user_session.get("subject",""),
            "standard": cl.user_session.get("standard","")
        }), tag="plan")

        await present_plan(plan)

        # Offer video
        ask_vid = cl.AskUserMessage(
            content="Create a short **explainer video with subtitles** from the plan? (yes/no)",
            timeout=300
        )
        vid_resp = await ask_vid.send()
        if normalize_reply(vid_resp).lower() in {"y","yes"}:
            avatar = cl.user_session.get("avatar_image") or DEFAULT_AVATAR
            # Build a brief narration
            lines = []
            for p in _as_list(plan.get("plans", [])):
                t = p.get("topic","Topic"); g = p.get("goals") or []
                lines.append(f"For {t}, goals include {', '.join(g) if g else 'concepts and practice'}.")
            narration = " ".join(lines) or "This plan covers your weak areas with focused practice."
            path = await make_video_with_subs(narration, avatar)
            try:
                vid_el = cl.Video(name=os.path.basename(path), path=path, display="inline")
            except Exception:
                vid_el = cl.File(name=os.path.basename(path), path=path, display="inline", mime="video/mp4")
            await cl.Message(content="Your narrated video is ready:", elements=[vid_el]).send()

        # Reset quiz waiting flag, then ask to restart
        cl.user_session.set("awaiting_quiz_answers", False)
        ask_restart = cl.AskUserMessage(
            content="Want to **start over** and try a different mode/subject? (yes/no)",
            timeout=300
        )
        res = await ask_restart.send()
        if normalize_reply(res).lower() in {"y","yes"}:
            await restart_ui()
        else:
            await cl.Message(content="Okay—type **/reset** anytime to begin again.").send()

# ───────────── CLI (optional) ─────────────
def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="Grading Assistant (CLI or Chainlit UI)")
    parser.add_argument("--mode", choices=["upload","quiz"], required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--standard", required=True)
    parser.add_argument("--qpdf", help="Question PDF (upload mode)")
    parser.add_argument("--apdf", help="Answer PDF (upload mode)")
    parser.add_argument("--make_video", action="store_true", help="Render explainer video with subtitles (upload mode)")
    args = parser.parse_args()

    avatar = pick_avatar_for_subject(args.subject)

    if args.mode == "upload":
        if not args.qpdf or not args.apdf:
            print("Provide --qpdf and --apdf for upload mode."); sys.exit(2)
        qtext = read_pdf_text(args.qpdf); atext = read_pdf_text(args.apdf)
        init: AppState = {"subject": args.subject, "standard": args.standard, "mode":"upload",
                          "q_text": qtext, "a_text": atext, "avatar_image": avatar}
        result = asyncio.run(GRAPH.ainvoke(init))
        graded_rows = result.get("graded", {}).get("graded", [])
        totals = result.get("graded", {}).get("totals", {"awarded":0,"out_of":0})
        print("Scores:")
        for r in graded_rows:
            print(f"- {r.get('id')} {r.get('topic')}: {r.get('awarded')}/{r.get('out_of')}")
        print(f"TOTAL: {totals.get('awarded',0)}/{totals.get('out_of',0)}")
        print("Gaps:", summarize_gaps_weighted(graded_rows))
        if args.make_video and result.get("plan_summary"):
            path = asyncio.run(make_video_with_subs(result["plan_summary"], avatar))
            print("Video written to:", path)

    else:
        init: AppState = {"subject": args.subject, "standard": args.standard, "mode":"quiz", "avatar_image": avatar}
        result = asyncio.run(GRAPH.ainvoke(init))
        qs = _coerce_questions(result.get("quiz", {}), 20)
        for i, q in enumerate(qs, 1):
            print(f"{i}. {q.get('text','')}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_main()
    else:
        print("Run with Chainlit:  chainlit run apps/app_grading_assistant.py")
        print("Or CLI: python apps/app_grading_assistant.py --help")
