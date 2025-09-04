# Grading Assistant (Agentic, LangGraph + Chainlit)

An end‑to‑end **grading & learning‑plan assistant** that can:

- **Extract** questions/answers from uploaded PDFs (question paper + student responses)
- **Grade** with a strict rubric and compute topic‑wise mastery
- **Plan** a weekly personalized course with references **and starter content**
- **Generate quizzes** (20 items) by subject/grade, collect answers in one message, and grade them
- **Render a narrated video** (static persona image + **subtitles**) of the plan
- **Optionally extend to lip‑sync video** using a companion module

All orchestration is done with **LangGraph**, with a clean **Chainlit** UI and a CLI.
No background services are required; everything runs locally once dependencies are installed.

---

## ✨ Highlights

- **Agents** (implemented as LangGraph nodes):
  1. **Extractor Agent** – parses the two PDFs and aligns questions & answers
  2. **Grader Agent** – rubric‑based grading with partial credit
  3. **Gap Analyzer Agent** – aggregates topic mastery
  4. **Planner Agent** – weekly personalized plan + references + starter content
  5. **Quiz Generator Agent** – produces 20 balanced questions with answer key & video points
  6. **Video Renderer** – TTS → static persona video → SRT subtitles (FFmpeg)
- **Two modes**: _Upload PDFs_ or _Generate Quiz_
- **Mock mode** for offline demos (no keys or network calls)
- **Chainlit UI** + **CLI** parity
- **Prompts are externalized** in `prompts/*.yaml` (easy to iterate without code changes)
- **personas are externalized** in `assets/*` (easy to iterate without code changes)

---

## 🧰 Requirements

- **Python** 3.10+ (tested on Windows 11 / Python 3.12)
- **FFmpeg** in your `PATH`
  - Windows: download from https://github.com/BtbN/FFmpeg-Builds/releases or https://www.gyan.dev/ffmpeg/builds/ and add `bin/` to PATH
  - macOS: `brew install ffmpeg`
  - Linux (Debian/Ubuntu): `sudo apt-get install ffmpeg`
      
    ```bash
      winget install --id Gyan.Dev.FFmpeg -e    
      After restart VS Code/termianl ffmpeg -version  
    ``` 
- **TTS** via `pyttsx3`
  - Windows uses SAPI5 automatically; macOS/Linux use platform defaults
- Azure OpenAI (for live mode) **or** set `MOCK_MODE = True` to run fully offline

---

## 🚀 Installation

> If you faced issues with `uv`, prefer standard `pip` on Windows.

### 1) Create a virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

**Option A: install the project (uses `pyproject.toml`)**

```bash
pip install uv
uv sync
```

**Option B: minimal direct install**

```bash
pip install chainlit langchain-openai pyttsx3 moviepy PyPDF2 pyyaml numpy tqdm jsonschema
```

### 3) Provide your model config

Create **`model_config.yaml`** at the project root:

```yaml
config:
  provider: "azure"
  azure_endpoint: "https://<your-resource-name>.openai.azure.com"
  api_key: "<YOUR_AZURE_OPENAI_KEY>"
  api_version: "2024-06-01"
  azure_deployment: "gpt-4.1"        # or your deployment name
  model: "gpt-4.1"                    # not always required by Azure, kept for clarity
  temperature: 0.2
  top_p: 1.0
  seed: 1
```

> To run **offline**, set `MOCK_MODE = True` inside `apps/app_grading_assistant.py` (top of the file). No keys needed.

### 4) (Optional) Prepare prompt files

You can start with the fallbacks, or create these in `prompts/`:



## ▶️ Running

### Chainlit UI

```bash
chainlit run apps/app_grading_assistant.py
chainlit run apps/app_avatar_viseme.py
# open http://localhost:8000
```

**Flow 1 — Upload PDFs**  
1) Choose subject & grade → Upload **Question PDF** then **Answer PDF**  
2) The app shows agent progress: Extractor → Grader → Gap Analyzer → Planner  
3) You get scores, feedback, mastery by topic, **personalized weekly plan** (with references & starter content)  
4) Optionally generate a **narrated video** (static persona image + subtitles)  
5) You’ll be prompted to **Start Over** (buttons re‑enabled on restart)

**Flow 2 — Quiz**  
1) Choose subject & grade → Generate **20‑item quiz**  
2) The UI prints only `"1. Question"` lines (minimal)  
3) **Send your answers in one message**, using any numbered style:  
   `1) ... 2) ... 3) ...` (or `1.`, `1:`, `1-`)  
4) The app grades, shows mastery, proposes a plan (with references & starter content), and offers video

### CLI (no UI)

Upload mode:
```bash
python apps/app_grading_assistant.py --mode upload --subject "Mathematics" --standard "Grade 10" --qpdf Q.pdf --apdf A.pdf --make_video
```

Quiz mode:
```bash
python apps/app_grading_assistant.py --mode quiz --subject "Science" --standard "Grade 9"
```

---

## 🧠 Architecture (LangGraph)

The main graph in `app_grading_assistant.py`:

```
router  --(mode=upload)-->  extract  ->  grade  ->  gaps  ->  plan  -> END
   \--(mode=quiz)------->  quiz --------------------------------------> END
```

- **Router**: dispatch based on chosen mode
- **Extractor**: returns `{questions[], answers[]}`
- **Grader**: returns `{graded[], totals{}}`
- **Gap Analyzer**: converts graded rows → mastery per topic
- **Planner**: builds weekly plan with references + starter content (+ summary for narration)
- **Quiz**: 20 questions + answer_key + video_points

All prompts are loaded from `prompts/*.yaml` (if missing, robust fallbacks are used).

---

## 🎬 Video Rendering

The main app produces a **static‑image narrated video with subtitles**:

1. TTS (`pyttsx3`) → WAV
2. Still image (persona) + audio → MP4 (FFmpeg)
3. SRT subtitles are synthesized from the narration text and burned (or embedded) with FFmpeg

**FFmpeg is required** and must be in PATH.

### Optional: Lip‑Sync Extension

`apps/visme_lipsync.py` demonstrates how to **import all core functionality** from the main app and add **lip‑sync rendering** (e.g., via Wav2Lip).

High‑level steps:
- Install Wav2Lip and download `Wav2Lip.pth`
- Ensure face image frames are prepared or use Wav2Lip’s inference pipeline
- Call the same narration & SRT generation from the core app, replacing the static video step with a lip‑sync render

> The extension is optional; the base app does not depend on Wav2Lip, keeping installation light.

---

## ⚙️ Configuration & Behavior

- **MOCK_MODE**: set at the top of `apps/app_grading_assistant.py`
  - `True` → no network calls; deterministic mock outputs
  - `False` → uses Azure OpenAI via `model_config.yaml`
- **Persona images**: selected by subject if available, else first image in `assets/`
- **Outputs**: videos and SRT files go to `outputs/`

---

## 🧩 Troubleshooting

- **FFmpeg error 4294967294 (Windows)**  
  Usually PATH or codec issue. Reinstall a full build (gyan.dev or BtbN) and ensure `ffmpeg.exe` is in PATH. Restart terminal.

- **Chainlit session / Start Over**  
  The app cleans session keys and re‑sends the initial buttons. If buttons seem disabled, type **/reset**.

- **`UserSession.clear` not found**  
  The app avoids that API; it deletes known keys and re‑calls `on_chat_start`.

- **TTS on Windows (pyttsx3)**  
  `comtypes` may log cache messages on first use; this is normal.

- **Azure 401/403/Timeouts**  
  Check `model_config.yaml` resource name, deployment name, API version, and key. Confirm the deployment is a Chat Completions model.

- **MoviePy warnings (`SyntaxWarning: invalid escape sequence '\P'`)**
  Benign and safe to ignore.

---

## 🔒 Privacy & Safety

- PDFs are processed locally; no files are uploaded by the app itself.  
- If `MOCK_MODE=False`, prompts and extracted text are sent to your Azure OpenAI endpoint.  
- Videos are generated locally and saved under `outputs/`.

---

## 📜 License

Proprietary / Internal demo. Update this section for your organization’s license policy.

---

## ❓ FAQ

**Q: Can I add more agents?**  
A: Yes—add a node, wire it in the LangGraph, and call it from the Chainlit flow.

**Q: Can I change the prompts without touching code?**  
A: Yes—edit YAML files in `prompts/`.

**Q: Can I use OpenAI’s public API instead of Azure?**  
A: The code currently targets Azure’s `langchain-openai` client. You can swap in `ChatOpenAI` with the same `call_json` helper.

**Q: How do I switch to lip‑sync?**  
A: Use `apps/visme_lipsync.py`. It imports the core narration/SRT helpers and replaces the “still image” step with a Wav2Lip render.
