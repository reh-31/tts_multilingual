from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
from pydub import AudioSegment
import os, uuid, threading, numpy as np

app = Flask(__name__)
OUTPUT_DIR = "outputs"
AUDIO_SAMPLES_DIR = "audio_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_SAMPLES_DIR, exist_ok=True)

# ── English model (Coqui VCTK) ───────────────────────────────────────────────
print("Loading English TTS model...")
tts_en = TTS(model_name="tts_models/en/vctk/vits")
print("English model loaded.")

SPEAKERS = {
    "doctor":   tts_en.speakers[0],
    "patient":  tts_en.speakers[10],
    "narrator": tts_en.speakers[5],
    "custom_1": tts_en.speakers[2],
    "custom_2": tts_en.speakers[8],
}

# ── indic-parler-tts (lazy loaded once on first Indic request) ───────────────
_indic_model = None
_indic_tokenizer = None
_indic_desc_tokenizer = None
_indic_device = None
_indic_lock = threading.Lock()

# Per-role voice descriptions for indic-parler-tts.
# These shape the prosody/gender/speed of each speaker.
# See: https://huggingface.co/ai4bharat/indic-parler-tts for recommended voices.
INDIC_VOICE_DESCRIPTIONS = {
    "doctor":   "Divya speaks in a calm, measured, professional tone with a clear and close recording.",
    "patient":  "Rajesh speaks in a slightly hesitant, conversational tone. The recording is very clear.",
    "narrator": "Leela speaks in a clear, neutral, and steady pace. The recording has no background noise.",
    "custom_1": "Meera speaks in a warm, friendly tone at a moderate pace. Recording is high quality.",
    "custom_2": "Arjun speaks in a confident, slightly fast-paced manner. Very clear recording.",
}
DEFAULT_INDIC_VOICE = "A speaker with a clear, neutral voice at a moderate pace. High quality recording."


def get_indic_model():
    """Lazy-load and cache the indic-parler-tts model (thread-safe)."""
    global _indic_model, _indic_tokenizer, _indic_desc_tokenizer, _indic_device
    with _indic_lock:
        if _indic_model is None:
            import torch
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer

            _indic_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading indic-parler-tts on {_indic_device}...")
            _indic_model = ParlerTTSForConditionalGeneration.from_pretrained(
                "ai4bharat/indic-parler-tts"
            ).to(_indic_device)
            _indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
            _indic_desc_tokenizer = AutoTokenizer.from_pretrained(
                _indic_model.config.text_encoder._name_or_path
            )
            print("indic-parler-tts loaded.")
    return _indic_model, _indic_tokenizer, _indic_desc_tokenizer, _indic_device


def synthesize_indic(text, role, lang, tmp_path):
    """Generate speech using indic-parler-tts and save to tmp_path."""
    import torch
    import soundfile as sf
    from pydub import AudioSegment

    model, tokenizer, desc_tokenizer, device = get_indic_model()

    description = INDIC_VOICE_DESCRIPTIONS.get(role.lower(), DEFAULT_INDIC_VOICE)

    desc_inputs = desc_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )

    audio_np = generation.cpu().numpy().squeeze()
    sample_rate = model.config.sampling_rate  # typically 44100

    # Save via soundfile then reload as pydub-compatible WAV
    sf_path = tmp_path.replace(".wav", "_sf.wav")
    sf.write(sf_path, audio_np, sample_rate)

    # Re-export to standard PCM WAV so pydub is happy
    seg = AudioSegment.from_wav(sf_path)
    seg.export(tmp_path, format="wav")
    if os.path.exists(sf_path):
        os.remove(sf_path)


jobs = {}
jobs_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
#  TTS Worker — routes per line to English or Indic engine
# ─────────────────────────────────────────────────────────────────────────────
def run_tts_job(job_id, dialogues, speaker_map):
    """
    dialogues: list of [role, text, lang]
      role  — one of SPEAKERS keys
      text  — the line to speak
      lang  — 'en' | 'hi' | 'mr' | 'gu'
    """
    tmp_files = []
    try:
        dialogues = [d for d in dialogues if (d[1] if isinstance(d, (list,tuple)) and len(d)>1 else d.get("text","")) ]
        with jobs_lock:
            jobs[job_id] = {"status": "processing", "progress": 0, "total": len(dialogues)}

        audio_segments = []
        for i, entry in enumerate(dialogues):
            # entry may be a list [role, text, lang] or a dict
            if isinstance(entry, dict):
                role = entry.get("role", "doctor")
                text = entry.get("text", "")
                lang = entry.get("lang", "en")
            else:
                entry = list(entry)
                role = entry[0] if len(entry) > 0 else "doctor"
                text = entry[1] if len(entry) > 1 else ""
                lang = entry[2] if len(entry) > 2 else "en"
            if not text:
                continue

            key = role.lower().replace(" ", "_")
            tmp_path = os.path.join(OUTPUT_DIR, f"{job_id}_line_{i}.wav")
            tmp_files.append(tmp_path)

            model_name = "Coqui VCTK" if lang == "en" else "Indic Parler TTS"
            with jobs_lock:
                jobs[job_id]["current_lang"]  = lang
                jobs[job_id]["current_model"] = model_name

            if lang == "en":
                speaker = speaker_map.get(key, tts_en.speakers[0])
                tts_en.tts_to_file(text=text, speaker=speaker, file_path=tmp_path)
            else:
                # hi / mr / gu → indic-parler-tts
                synthesize_indic(text, role=key, lang=lang, tmp_path=tmp_path)

            audio_segments.append(AudioSegment.from_wav(tmp_path))
            audio_segments.append(AudioSegment.silent(duration=450))

            with jobs_lock:
                jobs[job_id]["progress"] = i + 1

        final_path = os.path.join(OUTPUT_DIR, f"{job_id}_final.wav")
        sum(audio_segments).export(final_path, format="wav")
        with jobs_lock:
            jobs[job_id] = {"status": "done", "file": final_path}

    except Exception as e:
        with jobs_lock:
            jobs[job_id] = {"status": "error", "message": str(e)}
    finally:
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Embedded HTML  (language-aware UI)
# ─────────────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>VoiceCraft — Multilingual Voice Studio</title>
  <meta name="description" content="Create multilingual dialogue audio with AI voices in Hindi, Marathi, Gujarati and English."/>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500;1,600;1,700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #171412;
      --surface: rgba(255, 255, 255, 0.025);
      --surface-solid: #1c1816;
      --surface2: rgba(255, 255, 255, 0.04);
      --surface3: rgba(255, 255, 255, 0.08);
      --border: rgba(255, 255, 255, 0.06);
      --border-hover: rgba(255, 255, 255, 0.12);
      
      /* Studio Gold & Warm Bronze Color Scheme */
      --accent: #d29a59;
      --accent-hover: #e5ab6b;
      --accent-glow: rgba(210, 154, 89, 0.2);
      --accent-dim: rgba(210, 154, 89, 0.1);
      
      --text: #eadecf;
      --text-secondary: #999086;
      --muted: #665f57;
      
      --success: #63b175;
      --success-dim: rgba(99, 177, 117, 0.15);
      --error: #cf6666;
      --error-bg: rgba(207, 102, 102, 0.08);
      
      --radius: 18px;
      --radius-sm: 12px;
      --radius-xs: 8px;
    }
    
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg); color: var(--text);
      min-height: 100vh; overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* ── Ambient Background Glows ── */
    .ambient-bg {
      position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
    }
    .ambient-orb {
      position: absolute; border-radius: 50%; filter: blur(140px); opacity: 0.35;
      animation: orbFloat 25s ease-in-out infinite;
    }
    .ambient-orb:nth-child(1) {
      width: 600px; height: 600px; top: -15%; left: 10%;
      background: radial-gradient(circle, rgba(210, 154, 89, 0.12), transparent 75%);
      animation-delay: 0s;
    }
    .ambient-orb:nth-child(2) {
      width: 500px; height: 500px; bottom: -10%; right: 10%;
      background: radial-gradient(circle, rgba(210, 154, 89, 0.08), transparent 75%);
      animation-delay: -8s; animation-duration: 28s;
    }
    @keyframes orbFloat {
      0%, 100% { transform: translate(0, 0) scale(1); }
      33% { transform: translate(40px, -30px) scale(1.05); }
      66% { transform: translate(-35px, 25px) scale(0.95); }
    }

    /* ── Layout ── */
    .app-container {
      position: relative; z-index: 1;
      max-width: 960px; margin: 0 auto;
      padding: 60px 24px 120px;
    }

    /* ── Header / Hero ── */
    .hero {
      text-align: center; padding: 0 0 50px;
      position: relative;
    }
    
    /* Multilingual Studio Pill Badge */
    .hero-badge {
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(30, 26, 24, 0.6); 
      border: 1px solid rgba(210, 154, 89, 0.15);
      border-radius: 99px; padding: 6px 16px 6px 12px;
      font-size: 11px; font-weight: 600; color: var(--accent);
      letter-spacing: 0.12em; text-transform: uppercase;
      margin-bottom: 24px;
      animation: fadeUp 0.6s ease;
    }
    .hero-badge-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: #63b175;
      box-shadow: 0 0 8px rgba(99, 177, 117, 0.8);
      animation: pulseGreen 2.5s ease-in-out infinite;
    }
    @keyframes pulseGreen { 
      0%, 100% { opacity: 1; transform: scale(1); } 
      50% { opacity: 0.5; transform: scale(1.2); } 
    }

    /* Elegant Serif Branding */
    h1 {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: clamp(42px, 7vw, 68px);
      font-weight: 400; letter-spacing: -1px; line-height: 1.15;
      margin-bottom: 20px;
      color: #ffffff;
      animation: fadeUp 0.6s ease 0.1s both;
    }
    h1 .logo-serif {
      font-family: 'Playfair Display', Georgia, serif;
      font-style: normal;
      color: #ffffff;
    }
    h1 .logo-italic {
      font-family: 'Playfair Display', Georgia, serif;
      font-style: italic;
      font-weight: 400;
      color: var(--accent);
    }

    .hero-subtitle {
      font-size: 15px; color: var(--text-secondary); font-weight: 400;
      line-height: 1.7; max-width: 580px; margin: 0 auto 32px;
      animation: fadeUp 0.6s ease 0.2s both;
    }

    /* Language Badges matching layout */
    .lang-pills {
      display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;
      animation: fadeUp 0.6s ease 0.3s both;
    }
    .lang-pill {
      display: flex; align-items: center; gap: 8px;
      background: rgba(30, 26, 24, 0.4); 
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: 99px; padding: 7px 18px 7px 12px;
      font-size: 13px; font-weight: 500; color: var(--text-secondary);
      transition: all 0.3s ease;
    }
    .lang-pill:hover {
      border-color: rgba(210, 154, 89, 0.3); color: var(--text);
      background: rgba(210, 154, 89, 0.05);
      transform: translateY(-1px);
    }
    .lang-pill .pill-country {
      background: rgba(255, 255, 255, 0.06);
      border-radius: 4px; padding: 2px 6px;
      font-size: 10px; font-weight: 600; font-family: monospace;
      color: var(--accent);
    }

    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* ── Warm Gold Waveform ── */
    .waveform-container {
      display: flex; justify-content: center; gap: 4px;
      height: 38px; align-items: center; margin: 36px 0 0;
      animation: fadeUp 0.6s ease 0.4s both;
    }
    .wave-bar {
      width: 3px; border-radius: 99px;
      background: linear-gradient(180deg, var(--accent) 0%, rgba(210, 154, 89, 0.2) 100%);
      animation: waveAnim 1.4s ease-in-out infinite;
      opacity: 0.7;
    }
    @keyframes waveAnim {
      0%, 100% { height: 8px; opacity: 0.4; }
      50% { height: 34px; opacity: 0.9; }
    }

    /* ── Cards ── */
    .card {
      background: rgba(28, 24, 22, 0.45);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid rgba(255, 255, 255, 0.04);
      border-radius: var(--radius);
      padding: 32px;
      margin-bottom: 24px;
      transition: all 0.3s ease;
    }
    .card:hover { 
      border-color: rgba(210, 154, 89, 0.12); 
    }

    .card-header {
      display: flex; align-items: baseline; justify-content: space-between;
      margin-bottom: 24px;
    }
    .card-title-group {
      display: flex; flex-direction: column; gap: 4px;
    }
    .card-title {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 20px; font-weight: 500; letter-spacing: -0.2px;
      color: #ffffff;
    }
    .card-subtitle {
      font-size: 13px; color: var(--text-secondary);
    }
    .card-header-meta {
      font-size: 12px; color: var(--muted); font-weight: 500;
    }

    /* ── Cast (Speakers Grid) ── */
    .speakers-row {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
    }
    .speaker-chip {
      display: flex; align-items: center; gap: 12px;
      background: rgba(30, 26, 24, 0.6); 
      border: 1px solid rgba(255, 255, 255, 0.03);
      border-radius: var(--radius-sm); padding: 12px 16px;
      transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); 
      cursor: pointer;
      position: relative;
    }
    .speaker-chip:hover {
      border-color: rgba(210, 154, 89, 0.3);
      background: rgba(35, 30, 28, 0.8);
      transform: translateY(-2px);
    }
    .speaker-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 600; color: #171412;
      flex-shrink: 0;
    }
    .speaker-meta {
      display: flex; flex-direction: column; gap: 2px;
    }
    .speaker-chip-name {
      font-size: 13px; font-weight: 600; text-transform: capitalize;
      color: #ffffff;
    }
    .speaker-chip-desc {
      font-size: 11px; color: var(--text-secondary);
    }

    /* Active cast state visualization */
    .speaker-chip.active {
      border-color: var(--accent);
      background: rgba(210, 154, 89, 0.05);
      box-shadow: 0 4px 16px rgba(210, 154, 89, 0.05);
    }

    /* ── Bulk Script Input ── */
    .bulk-textarea {
      width: 100%;
      background: rgba(20, 18, 16, 0.6); 
      border: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text); font-family: 'Inter', sans-serif; font-size: 13px;
      border-radius: var(--radius-sm); padding: 18px;
      outline: none; transition: all 0.3s ease;
      resize: vertical; min-height: 120px; line-height: 1.7;
    }
    .bulk-textarea::placeholder { color: var(--muted); }
    .bulk-textarea:focus {
      border-color: rgba(210, 154, 89, 0.4);
      box-shadow: 0 0 0 3px rgba(210, 154, 89, 0.1);
    }

    /* ── Dialogue Rows ── */
    #dialogue-list { display: flex; flex-direction: column; gap: 12px; }

    .dialogue-row {
      display: grid;
      grid-template-columns: 140px 130px 1fr 40px;
      gap: 12px; align-items: stretch;
      animation: rowSlideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes rowSlideIn {
      from { opacity: 0; transform: translateY(-8px) scale(0.99); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    .col-headers {
      display: grid;
      grid-template-columns: 140px 130px 1fr 40px;
      gap: 12px; margin-bottom: 8px; padding: 0 4px;
    }
    .col-header {
      font-size: 10px; font-weight: 600; color: var(--muted);
      letter-spacing: 0.1em; text-transform: uppercase;
    }

    select, .dialogue-row textarea {
      background: rgba(20, 18, 16, 0.6); 
      border: 1px solid rgba(255, 255, 255, 0.04);
      color: var(--text); font-family: 'Inter', sans-serif; font-size: 13px;
      border-radius: var(--radius-xs); padding: 12px 14px;
      outline: none; transition: all 0.25s ease; width: 100%;
    }
    select:focus, .dialogue-row textarea:focus {
      border-color: rgba(210, 154, 89, 0.4);
      box-shadow: 0 0 0 3px rgba(210, 154, 89, 0.10);
    }
    select:hover, .dialogue-row textarea:hover { border-color: rgba(255, 255, 255, 0.12); }

    select {
      cursor: pointer; appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23665f57' d='M5 6L0 0h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 14px center; padding-right: 32px;
    }
    select.lang-en { border-left: 3px solid #38bdf8; }
    select.lang-hi, select.lang-sa, select.lang-ne, select.lang-doi, select.lang-mai, select.lang-brx { border-left: 3px solid #f97316; }
    select.lang-mr, select.lang-ml, select.lang-ta, select.lang-te, select.lang-kn { border-left: 3px solid #2dd4bf; }
    select.lang-gu, select.lang-pa, select.lang-or, select.lang-as { border-left: 3px solid #fbbf24; }
    select.lang-ur, select.lang-sd, select.lang-kok, select.lang-mni, select.lang-sat, select.lang-bn { border-left: 3px solid #f472b6; }

    .dialogue-row textarea {
      resize: none; height: 50px; line-height: 1.6;
    }

    .delete-btn {
      background: transparent; border: 1px solid rgba(255, 255, 255, 0.04); color: var(--muted);
      cursor: pointer; border-radius: var(--radius-xs); width: 40px; font-size: 14px;
      transition: all 0.25s ease; display: flex; align-items: center; justify-content: center;
    }
    .delete-btn:hover {
      border-color: var(--error); color: var(--error);
      background: var(--error-bg);
      transform: scale(1.05);
    }

    /* ── Action buttons ── */
    .action-bar {
      display: flex; gap: 12px; flex-wrap: wrap;
      margin-top: 24px;
    }
    .btn {
      font-family: 'Inter', sans-serif;
      font-weight: 500; font-size: 13px; 
      border: none; border-radius: var(--radius-sm);
      padding: 12px 24px; cursor: pointer;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      display: inline-flex; align-items: center; gap: 8px;
      position: relative; overflow: hidden;
    }

    .btn-ghost {
      background: rgba(30, 26, 24, 0.6); color: var(--text-secondary);
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .btn-ghost:hover {
      border-color: rgba(210, 154, 89, 0.3); color: var(--text);
      background: rgba(210, 154, 89, 0.06);
      transform: translateY(-1px);
    }
    .btn-primary {
      background: var(--accent);
      color: #171412;
      font-weight: 600;
    }
    .btn-primary:hover {
      transform: translateY(-2px);
      background: var(--accent-hover);
      box-shadow: 0 6px 20px rgba(210, 154, 89, 0.25);
    }
    .btn-primary:active { transform: translateY(0); }
    .btn-primary:disabled {
      opacity: 0.35; cursor: not-allowed; transform: none;
      box-shadow: none;
    }
    .btn-success {
      background: #63b175;
      color: #112d18;
      font-weight: 600;
    }
    .btn-success:hover {
      transform: translateY(-2px);
      background: #73c285;
      box-shadow: 0 6px 18px rgba(99, 177, 117, 0.3);
    }

    /* ── Progress Indicators ── */
    #status-box {
      display: none;
      background: rgba(28, 24, 22, 0.45); backdrop-filter: blur(24px);
      border: 1px solid rgba(255, 255, 255, 0.04); border-radius: var(--radius);
      padding: 32px; margin-bottom: 24px;
      animation: fadeUp 0.4s ease;
    }
    .progress-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 16px;
    }
    .progress-title {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 16px; font-weight: 500;
      display: flex; align-items: center; gap: 10px;
    }
    .progress-spinner {
      width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.06);
      border-top-color: var(--accent); border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .progress-count {
      font-size: 13px; font-weight: 600; color: var(--accent);
      font-variant-numeric: tabular-nums;
    }
    .progress-bar-bg {
      background: rgba(20, 18, 16, 0.6); border-radius: 99px; height: 8px;
      overflow: hidden; position: relative;
    }
    .progress-bar-fill {
      height: 100%; border-radius: 99px;
      background: linear-gradient(90deg, #d29a59, #e5ab6b, #d29a59);
      background-size: 200% 100%;
      animation: progressGlow 2.5s linear infinite;
      transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1);
      width: 0%;
    }
    @keyframes progressGlow {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    .status-msg {
      margin-top: 14px; font-size: 13px; color: var(--text-secondary);
    }
    .model-info {
      display: none; margin-top: 14px; padding: 10px 16px;
      border-radius: var(--radius-xs);
      background: rgba(20, 18, 16, 0.4); border: 1px solid rgba(255, 255, 255, 0.04);
      font-size: 12px; display: flex; gap: 10px; align-items: center;
    }
    .mi-label { color: var(--muted); font-weight: 500; }
    .mi-lang-badge {
      font-weight: 700; padding: 3px 10px; border-radius: 6px;
      font-size: 10px; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .mi-lang-en {
      background: rgba(56,189,248,0.08); color: #38bdf8;
      border: 1px solid rgba(56,189,248,0.15);
    }
    .mi-lang-indic {
      background: rgba(210, 154, 89, 0.08); color: var(--accent);
      border: 1px solid rgba(210, 154, 89, 0.15);
    }
    .mi-arrow { color: var(--muted); }
    .mi-model-name {
      color: var(--text); font-weight: 600;
    }

    /* ── Audio Results ── */
    #audio-section {
      display: none;
      background: rgba(28, 24, 22, 0.45); backdrop-filter: blur(24px);
      border: 1px solid rgba(255, 255, 255, 0.04); border-radius: var(--radius);
      padding: 32px;
      animation: fadeUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .audio-header {
      display: flex; align-items: center; gap: 12px; margin-bottom: 24px;
    }
    .audio-live-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 12px rgba(99, 177, 117, 0.6);
      animation: pulseGreen 2s ease-in-out infinite;
    }
    .audio-title {
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 20px; font-weight: 500;
    }

    audio {
      width: 100%; margin-bottom: 24px; border-radius: var(--radius-xs);
      accent-color: var(--accent);
    }

    .download-row {
      display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
      padding: 16px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.04); margin-bottom: 24px;
    }
    .download-input {
      background: rgba(20, 18, 16, 0.6) !important; border: 1px solid rgba(255, 255, 255, 0.04) !important;
      color: var(--text) !important;
      font-family: 'Inter', sans-serif !important; font-size: 13px !important;
      border-radius: var(--radius-xs) !important;
      padding: 10px 14px !important; outline: none !important; width: 200px !important;
      transition: all 0.25s ease !important;
    }
    .download-input:focus {
      border-color: rgba(210, 154, 89, 0.4) !important;
      box-shadow: 0 0 0 3px rgba(210, 154, 89, 0.1) !important;
    }
    .download-ext { color: var(--muted); font-size: 13px; font-weight: 500; }

    /* ── Transcript Preview ── */
    .transcript-section {
      padding-top: 4px;
    }
    .transcript-label {
      font-size: 10px; font-weight: 600; color: var(--muted);
      letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 16px;
    }
    .transcript {
      display: flex; flex-direction: column; gap: 12px;
    }
    .transcript-line {
      display: flex; gap: 12px; align-items: baseline;
      padding: 10px 16px; border-radius: var(--radius-xs);
      background: rgba(20, 18, 16, 0.4);
      border: 1px solid rgba(255, 255, 255, 0.02);
      transition: all 0.2s;
    }
    .transcript-line:hover { 
      background: rgba(30, 26, 24, 0.6); 
      border-color: rgba(255, 255, 255, 0.04);
    }
    .t-role {
      font-family: 'Playfair Display', Georgia, serif;
      font-weight: 600; min-width: 80px; text-transform: capitalize;
      font-size: 13px;
    }
    .t-lang {
      font-size: 11px; color: var(--muted); min-width: 30px;
      font-weight: 500;
    }
    .t-text { color: var(--text-secondary); font-size: 13px; line-height: 1.6; }

    /* ── Error Notification ── */
    .error-msg {
      background: var(--error-bg);
      border: 1px solid rgba(207, 102, 102, 0.2);
      color: var(--error);
      border-radius: var(--radius-sm);
      padding: 14px 20px; font-size: 13px; font-weight: 500;
      margin-top: 18px; display: none;
      animation: shake 0.4s ease;
    }
    @keyframes shake {
      0%, 100% { transform: translateX(0); }
      25% { transform: translateX(-4px); }
      75% { transform: translateX(4px); }
    }

    /* ── Responsive breakpoints ── */
    @media (max-width: 640px) {
      .app-container { padding: 32px 16px 80px; }
      .hero { padding: 16px 0 32px; }
      .dialogue-row { grid-template-columns: 1fr 1fr; }
      .dialogue-row textarea { grid-column: 1 / -1; }
      .dialogue-row .delete-btn { grid-column: 2; justify-self: end; }
      .col-headers { display: none; }
      .card { padding: 24px 16px; }
    }
  </style>
</head>
<body>
  <div class="ambient-bg">
    <div class="ambient-orb"></div>
    <div class="ambient-orb"></div>
  </div>

  <div class="app-container">
    <!-- Hero / Title Header -->
    <div class="hero">
      <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        <span>Multilingual Voice Studio</span>
      </div>
      <h1>
        <span class="logo-serif">Voice</span><span class="logo-italic">Craft</span>
      </h1>
      <p class="hero-subtitle">
        Write a conversation, cast a voice for each speaker, and hear it come alive — blending English with 20+ Indic languages including Hindi, Marathi, Gujarati, Bengali, Tamil, Telugu &amp; more in one natural exchange.
      </p>
      <div class="lang-pills">
        <span class="lang-pill"><span class="pill-country">IN</span> Hindi हिंदी</span>
        <span class="lang-pill"><span class="pill-country">IN</span> Bengali বাংলা</span>
        <span class="lang-pill"><span class="pill-country">IN</span> Marathi मराठी</span>
        <span class="lang-pill"><span class="pill-country">IN</span> Gujarati ગુજરાતી</span>
        <span class="lang-pill"><span class="pill-country">IN</span> Tamil தமிழ்</span>
        <span class="lang-pill"><span class="pill-country">IN</span> Telugu తెలుగు</span>
        <span class="lang-pill"><span class="pill-country">GB</span> English EN</span>
        <span class="lang-pill" style="border-color: rgba(210, 154, 89, 0.25); color: var(--accent);"><span class="pill-country">+14</span> More</span>
      </div>
      <div class="waveform-container" id="waveform"></div>
    </div>

    <!-- The Cast Selection Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <div class="card-title">The cast</div>
          <div class="card-subtitle">Choose who's speaking. Tap a voice to make it active.</div>
        </div>
        <div class="card-header-meta" id="selected-cast-meta">2 selected</div>
      </div>
      <div class="speakers-row" id="speakers-grid"></div>
    </div>

    <!-- Bulk Script Input Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <div class="card-title">Quick Script</div>
          <div class="card-subtitle">Write dialogues in bulk with speaker prefix tags.</div>
        </div>
      </div>
      <textarea id="bulk-input" class="bulk-textarea" placeholder="Doctor [hi]: नमस्ते, आप कैसे हैं?&#10;Patient [en]: I have a headache.&#10;Doctor [mr]: तुम्हाला ताप आहे का?&#10;Patient [gu]: ના, ફક્ત ઉબકા."></textarea>
      <div class="action-bar">
        <button class="btn btn-ghost" onclick="convertBulkToDialogue()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 14l-7 7m0 0l-7-7m7 7V3"/></svg>
          Convert to Dialogue
        </button>
      </div>
    </div>

    <!-- Dialogue Lines Builder Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-title-group">
          <div class="card-title">Your script</div>
          <div class="card-subtitle">Create and reorder dialogue turn-by-turn.</div>
        </div>
        <div class="card-header-meta" id="line-count">0 lines</div>
      </div>
      <div class="col-headers">
        <span class="col-header">Speaker</span>
        <span class="col-header">Language</span>
        <span class="col-header">Dialogue</span>
        <span class="col-header"></span>
      </div>
      <div id="dialogue-list"></div>

      <div class="action-bar">
        <button class="btn btn-ghost" onclick="addRow()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M12 5v14m-7-7h14"/></svg>
          Add Line
        </button>
        <button class="btn btn-ghost" onclick="loadExample()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/></svg>
          Load Example
        </button>
        <button class="btn btn-primary" id="gen-btn" onclick="generateAudio()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/></svg>
          Generate Audio
        </button>
      </div>
    </div>

    <div id="error-msg" class="error-msg"></div>

    <!-- Status progress panel -->
    <div id="status-box">
      <div class="progress-header">
        <div class="progress-title">
          <div class="progress-spinner"></div>
          Synthesizing voices...
        </div>
        <span class="progress-count" id="progress-count">0 / 0</span>
      </div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="progress-fill"></div></div>
      <div class="status-msg" id="status-msg">Starting...</div>
      <div class="model-info" id="model-info">
        <span class="mi-label">Engine:</span>
        <span class="mi-lang-badge" id="mi-lang">—</span>
        <span class="mi-arrow">→</span>
        <span class="mi-model-name" id="mi-model">—</span>
      </div>
    </div>

    <!-- Generated Audio Output -->
    <div id="audio-section">
      <div class="audio-header">
        <div class="audio-live-dot"></div>
        <div class="audio-title">Your Audio is Ready</div>
      </div>
      <audio id="audio-player" controls></audio>
      <div class="download-row">
        <input type="text" id="download-name" class="download-input" placeholder="filename" value="conversation"/>
        <span class="download-ext">.wav</span>
        <button class="btn btn-success" onclick="downloadAudio()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
          Download WAV
        </button>
        <button class="btn btn-ghost" onclick="resetAll()">
          <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M1 4v6h6M23 20v-6h-6"/><path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15"/></svg>
          New
        </button>
      </div>
      <div class="transcript-section">
        <div class="transcript-label">Transcript</div>
        <div class="transcript" id="transcript-preview"></div>
      </div>
    </div>
  </div>

<script>
  const SPEAKER_NAMES = __SPEAKER_NAMES__;
  const SPEAKER_COLORS = ['#f97316','#2dd4bf','#38bdf8','#f472b6','#a3e635'];
  
  // Custom descriptions to match the mockup styling
  const SPEAKER_DESCS = {
    'doctor': 'Warm, measured',
    'patient': 'Soft, natural',
    'narrator': 'Calm, neutral',
    'custom_1': 'Your voice',
    'custom_2': 'Your voice'
  };

  const SPEAKER_INITIALS_BG = [
    '#f97316',
    '#2dd4bf',
    '#38bdf8',
    '#f472b6',
    '#a3e635',
  ];

  const LANGUAGES = [
    { code: 'en', label: 'English 🇬🇧' },
    { code: 'hi', label: 'Hindi हिंदी 🇮🇳' },
    { code: 'bn', label: 'Bengali বাংলা 🇮🇳' },
    { code: 'mr', label: 'Marathi मराठी 🇮🇳' },
    { code: 'gu', label: 'Gujarati ગુજરાતી 🇮🇳' },
    { code: 'ta', label: 'Tamil தமிழ் 🇮🇳' },
    { code: 'te', label: 'Telugu తెలుగు 🇮🇳' },
    { code: 'kn', label: 'Kannada ಕನ್ನಡ 🇮🇳' },
    { code: 'ml', label: 'Malayalam മലയാളം 🇮🇳' },
    { code: 'ur', label: 'Urdu اردو 🇮🇳' },
    { code: 'pa', label: 'Punjabi ਪੰਜਾਬੀ 🇮🇳' },
    { code: 'or', label: 'Odia ଓଡ଼ିଆ 🇮🇳' },
    { code: 'as', label: 'Assamese অসমীয়া 🇮🇳' },
    { code: 'sa', label: 'Sanskrit संस्कृतम् 🇮🇳' },
    { code: 'ne', label: 'Nepali नेपाली 🇳🇵' },
    { code: 'sd', label: 'Sindhi سنڌي 🇮🇳' },
    { code: 'kok', label: 'Konkani कोंकणी 🇮🇳' },
    { code: 'doi', label: 'Dogri डोगरी 🇮🇳' },
    { code: 'mai', label: 'Maithili मैथिली 🇮🇳' },
    { code: 'brx', label: 'Bodo बड़ो 🇮🇳' },
    { code: 'mni', label: 'Manipuri মৈতৈলোন 🇮🇳' },
    { code: 'sat', label: 'Santali ᱥᱟᱱᱛᱟᱲᱤ 🇮🇳' }
  ];
  const LANG_FLAGS = { en: '🇬🇧', hi: '🇮🇳hi', mr: '🇮🇳mr', gu: '🇮🇳gu', bn: '🇮🇳bn', ta: '🇮🇳ta', te: '🇮🇳te', kn: '🇮🇳kn', ml: '🇮🇳ml', ur: '🇮🇳ur', pa: '🇮🇳pa', or: '🇮🇳or', as: '🇮🇳as', sa: '🇮🇳sa', ne: '🇳🇵ne', sd: '🇮🇳sd', kok: '🇮🇳kok', doi: '🇮🇳doi', mai: '🇮🇳mai', brx: '🇮🇳brx', mni: '🇮🇳mni', sat: '🇮🇳sat' };

  const BULK_LANG_MAP = { 
    en:'en', hi:'hi', mr:'mr', gu:'gu', bn:'bn', ta:'ta', te:'te', kn:'kn', ml:'ml', ur:'ur', pa:'pa', or:'or', as:'as', sa:'sa', ne:'ne', sd:'sd', kok:'kok', doi:'doi', mai:'mai', brx:'brx', mni:'mni', sat:'sat',
    english:'en', hindi:'hi', marathi:'mr', gujarati:'gu', bengali:'bn', tamil:'ta', telugu:'te', kannada:'kn', malayalam:'ml', urdu:'ur', punjabi:'pa', odia:'or', assamese:'as', sanskrit:'sa', nepali:'ne', sindhi:'sd', konkani:'kok', dogri:'doi', maithili:'mai', bodo:'brx', manipuri:'mni', santali:'sat'
  };

  let rows = [];
  let rowCounter = 0;
  let pollingTimer = null;
  let currentJobId = null;

  // Track active cast selection status
  let activeCast = new Set(['doctor', 'patient']);

  // ── Waveform bars ──
  (function initWaveform() {
    const container = document.getElementById('waveform');
    for (let i = 0; i < 40; i++) {
      const bar = document.createElement('div');
      bar.className = 'wave-bar';
      bar.style.animationDelay = `${(i * 0.06)}s`;
      bar.style.height = '8px';
      container.appendChild(bar);
    }
  })();

  function updateLineCount() {
    const el = document.getElementById('line-count');
    if (el) el.textContent = rows.length + ' line' + (rows.length !== 1 ? 's' : '');
  }

  function updateCastMeta() {
    const el = document.getElementById('selected-cast-meta');
    if (el) el.textContent = activeCast.size + ' selected';
  }

  function toggleCast(name) {
    if (activeCast.has(name)) {
      if (activeCast.size > 1) { // keep at least 1
        activeCast.delete(name);
      }
    } else {
      activeCast.add(name);
    }
    renderSpeakers();
    updateCastMeta();
  }

  function renderSpeakers() {
    const grid = document.getElementById('speakers-grid');
    grid.innerHTML = '';
    SPEAKER_NAMES.forEach((name, i) => {
      const chip = document.createElement('div');
      const cleanName = name.toLowerCase();
      const isActive = activeCast.has(cleanName);
      chip.className = `speaker-chip ${isActive ? 'active' : ''}`;
      chip.onclick = () => toggleCast(cleanName);
      
      const initials = name.replace(/_/g,' ').split(' ').map(w => w[0]).join('').toUpperCase().slice(0,2);
      const desc = SPEAKER_DESCS[cleanName] || 'Custom voice';
      
      chip.innerHTML = `
        <div class="speaker-avatar" style="background:${SPEAKER_INITIALS_BG[i % SPEAKER_INITIALS_BG.length]}">${initials}</div>
        <div class="speaker-meta">
          <span class="speaker-chip-name">${name.replace(/_/g,' ')}</span>
          <span class="speaker-chip-desc">${desc}</span>
        </div>`;
      grid.appendChild(chip);
    });
  }

  function buildLangSelect(selectedLang, rowId) {
    const opts = LANGUAGES.map(l =>
      `<option value="${l.code}" ${l.code === selectedLang ? 'selected' : ''}>${l.label}</option>`
    ).join('');
    return `<select id="lang-${rowId}" class="lang-${selectedLang}" onchange="onLangChange(this)">${opts}</select>`;
  }

  function onLangChange(sel) {
    sel.className = 'lang-' + sel.value;
  }

  function addRow(role = '', text = '', lang = 'en') {
    const id = `row-${Date.now()}-${rowCounter++}`;
    rows.push(id);
    const list = document.getElementById('dialogue-list');
    const div = document.createElement('div');
    div.className = 'dialogue-row';
    div.id = id;
    
    // Default speaker to one of active cast members
    let defaultSpeaker = role;
    if (!defaultSpeaker) {
      defaultSpeaker = Array.from(activeCast)[0] || SPEAKER_NAMES[0];
    }
    
    const spkOpts = SPEAKER_NAMES.map(n =>
      `<option value="${n}" ${n === defaultSpeaker ? 'selected' : ''}>${n.replace(/_/g,' ')}</option>`
    ).join('');
    div.innerHTML = `
      <select id="role-${id}">${spkOpts}</select>
      ${buildLangSelect(lang, id)}
      <textarea id="text-${id}" placeholder="Type dialogue here...">${text}</textarea>
      <button class="delete-btn" onclick="removeRow('${id}')">✕</button>`;
    list.appendChild(div);
    updateLineCount();
  }

  function removeRow(id) {
    const el = document.getElementById(id);
    if (el) {
      el.style.opacity = '0'; el.style.transform = 'translateX(16px) scale(0.98)';
      el.style.transition = 'all 0.25s ease';
      setTimeout(() => { el.remove(); rows = rows.filter(r => r !== id); updateLineCount(); }, 250);
    }
  }

  function loadExample() {
    document.getElementById('dialogue-list').innerHTML = '';
    rows = [];
    [
      ['doctor',  'Good morning. What brings you here today?',          'en'],
      ['patient', 'मुझे दो दिनों से पेट में दर्द हो रहा है।',           'hi'],
      ['doctor',  'तुम्हाला ताप किंवा उलट्या होत आहेत का?',            'mr'],
      ['patient', 'ના, પણ મને ઉબકા આવે છે.',                           'gu'],
      ['doctor',  'Understood. I will prescribe some medication.',       'en'],
      ['patient', 'धन्यवाद डॉक्टर। कितने दिन लेनी है दवाई?',           'hi'],
      ['doctor',  'पाच दिवस घ्या आणि बरे न वाटल्यास परत या.',          'mr'],
    ].forEach(([r, t, l]) => addRow(r, t, l));
  }

  const HINDI_SPEAKER_ALIASES = {
    'doctor': 'doctor', 'patient': 'patient', 'narrator': 'narrator',
    'custom_1': 'custom_1', 'custom_2': 'custom_2',
    'डॉक्टर': 'doctor', 'मरीज': 'patient', 'मरीज़': 'patient', 'वर्णनकर्ता': 'narrator',
    'रुग्ण': 'patient', 'निवेदक': 'narrator',
    'ડૉક્ટર': 'doctor', 'દર્દી': 'patient', 'નિવેદક': 'narrator',
  };

  function _detectLang(text, prefix) {
    const all = text + ' ' + prefix;
    const codes = [...all].map(c => c.charCodeAt(0));
    if (codes.some(c => c >= 0x0a80 && c <= 0x0aff)) return 'gu';
    if (codes.some(c => c >= 0x0a00 && c <= 0x0a7f)) return 'pa';
    if (codes.some(c => c >= 0x0980 && c <= 0x09ff)) return 'bn';
    if (codes.some(c => c >= 0x0c00 && c <= 0x0c7f)) return 'te';
    if (codes.some(c => c >= 0x0c80 && c <= 0x0cff)) return 'kn';
    if (codes.some(c => c >= 0x0d00 && c <= 0x0d7f)) return 'ml';
    if (codes.some(c => c >= 0x0b80 && c <= 0x0bff)) return 'ta';
    if (codes.some(c => c >= 0x0b00 && c <= 0x0b7f)) return 'or';
    if (codes.some(c => c >= 0x0600 && c <= 0x06ff)) return 'ur';
    if (codes.some(c => c >= 0x0900 && c <= 0x097f)) return 'hi';
    return 'en';
  }

  function convertBulkToDialogue() {
    const raw = document.getElementById('bulk-input')?.value || '';
    const lines = raw.replace(/：/g, ':').split(/\\r?\\n/);
    document.getElementById('dialogue-list').innerHTML = '';
    rows = [];

    const speakerLookup = new Map(SPEAKER_NAMES.map(n => [n.toLowerCase(), n]));
    let added = 0;

    for (const lineRaw of lines) {
      const line = lineRaw.trim();
      if (!line) continue;
      const colonIdx = line.indexOf(':');
      if (colonIdx <= 0) continue;

      const prefix = line.slice(0, colonIdx).trim();
      const text = line.slice(colonIdx + 1).trim();
      if (!text) continue;

      const langMatch = prefix.match(/\\[([a-z]+)\\]\\s*$/i);
      let langRaw = langMatch ? langMatch[1].toLowerCase() : null;
      const speakerRaw = prefix.replace(/\\[[^\\]]+\\]/, '').trim().toLowerCase().replace(/\\s+/g, '_');

      const prefixClean = prefix.replace(/\\[[^\\]]+\\]/, '').trim();
      const speaker = HINDI_SPEAKER_ALIASES[prefixClean]
                   || HINDI_SPEAKER_ALIASES[prefixClean.toLowerCase()]
                   || speakerLookup.get(speakerRaw)
                   || (added % 2 === 0 ? 'doctor' : 'patient');

      if (!langRaw) { langRaw = _detectLang(text, prefix); }
      const lang = BULK_LANG_MAP[langRaw] || 'en';

      addRow(speaker, text, lang);
      
      // Auto-add parsed speaker to active cast
      activeCast.add(speaker.toLowerCase());
      
      added++;
    }

    if (added === 0) {
      showError('No valid lines found. Use format: Speaker [lang]: text');
      return;
    }
    
    renderSpeakers();
    updateCastMeta();
    hideError();
  }

  async function generateAudio() {
    const dialogues = rows.map(id => [
      document.getElementById(`role-${id}`)?.value,
      document.getElementById(`text-${id}`)?.value?.trim(),
      document.getElementById(`lang-${id}`)?.value || 'en',
    ]).filter(([r, t]) => r && t);

    if (!dialogues.length) { showError('Please add at least one dialogue line.'); return; }
    hideError();
    document.getElementById('gen-btn').disabled = true;
    document.getElementById('audio-section').style.display = 'none';
    document.getElementById('status-box').style.display = 'block';
    document.getElementById('status-msg').textContent = 'Sending to server...';
    setProgress(0, dialogues.length);

    try {
      const res = await fetch('/api/generate', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ dialogues })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      pollStatus(data.job_id, dialogues);
    } catch(e) {
      showError('Failed to start: ' + e.message);
      document.getElementById('gen-btn').disabled = false;
      document.getElementById('status-box').style.display = 'none';
    }
  }

  function pollStatus(jobId, dialogues) {
    pollingTimer = setInterval(async () => {
      const data = await (await fetch(`/api/status/${jobId}`)).json();
      if (data.status === 'processing') {
        setProgress(data.progress, data.total);
        document.getElementById('status-msg').textContent = `Rendering line ${data.progress} of ${data.total}...`;
        if (data.current_lang) {
          const lang = data.current_lang;
          const isIndic = lang !== 'en';
          const langEl  = document.getElementById('mi-lang');
          const modelEl = document.getElementById('mi-model');
          const infoEl  = document.getElementById('model-info');
          langEl.textContent = lang.toUpperCase();
          langEl.className   = 'mi-lang-badge ' + (isIndic ? 'mi-lang-indic' : 'mi-lang-en');
          modelEl.textContent = data.current_model || (isIndic ? 'Indic Parler TTS' : 'Coqui VCTK');
          infoEl.style.display = 'flex';
        }
      } else if (data.status === 'done') {
        clearInterval(pollingTimer);
        setProgress(data.total||1, data.total||1);
        document.getElementById('status-msg').textContent = '✓ All done!';
        setTimeout(() => showResult(jobId, dialogues), 500);
      } else if (data.status === 'error') {
        clearInterval(pollingTimer);
        showError('TTS error: ' + data.message);
        document.getElementById('gen-btn').disabled = false;
        document.getElementById('status-box').style.display = 'none';
      }
    }, 1200);
  }

  function setProgress(done, total) {
    document.getElementById('progress-fill').style.width = (total ? done/total*100 : 0) + '%';
    document.getElementById('progress-count').textContent = `${done} / ${total}`;
  }

  function downloadAudio() {
    if (!currentJobId) return;
    const raw = (document.getElementById('download-name')?.value || 'conversation').trim();
    const filename = raw.endsWith('.wav') ? raw : raw + '.wav';
    const a = document.createElement('a');
    a.href = `/api/download/${currentJobId}?filename=${encodeURIComponent(filename)}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }



  function showResult(jobId, dialogues) {
    currentJobId = jobId;
    document.getElementById('status-box').style.display = 'none';
    document.getElementById('gen-btn').disabled = false;
    document.getElementById('audio-section').style.display = 'block';
    const url = `/api/download/${jobId}?preview=1&t=${Date.now()}`;
    document.getElementById('audio-player').src = url;
    const tp = document.getElementById('transcript-preview');
    tp.innerHTML = '';
    dialogues.forEach(([role, text, lang]) => {
      const color = SPEAKER_COLORS[SPEAKER_NAMES.indexOf(role) % SPEAKER_COLORS.length] || '#a1a1b5';
      const line = document.createElement('div');
      line.className = 'transcript-line';
      line.innerHTML = `
        <span class="t-role" style="color:${color}">${role.replace(/_/g,' ')}</span>
        <span class="t-lang">[${lang||'en'}]</span>
        <span class="t-text">${text}</span>`;
      tp.appendChild(line);
    });
  }

  function resetAll() {
    currentJobId = null;
    document.getElementById('audio-section').style.display = 'none';
    document.getElementById('status-box').style.display = 'none';
    document.getElementById('model-info').style.display = 'none';
    hideError();
  }
  function showError(m) { const e=document.getElementById('error-msg'); e.textContent='⚠ '+m; e.style.display='block'; }
  function hideError() { document.getElementById('error-msg').style.display='none'; }

  renderSpeakers();
  loadExample();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    import json
    speaker_names = list(SPEAKERS.keys())
    page = HTML.replace("__SPEAKER_NAMES__", json.dumps(speaker_names))
    return page, 200, {"Content-Type": "text/html"}


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    dialogues = data.get("dialogues", [])   # [[role, text, lang], ...]
    if not dialogues:
        return jsonify({"error": "No dialogues provided"}), 400
    speaker_map = {k: v for k, v in SPEAKERS.items()}
    job_id = str(uuid.uuid4())[:8]
    threading.Thread(
        target=run_tts_job, args=(job_id, dialogues, speaker_map), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    return jsonify(job or {"status": "not_found"}), 200 if job else 404


@app.route("/api/download/<job_id>")
def download(job_id):
    import shutil
    with jobs_lock:
        job = jobs.get(job_id)

    # Fallback: reconstruct job from the final file if server was restarted
    if not job or job.get("status") != "done":
        fallback_path = os.path.join(OUTPUT_DIR, f"{job_id}_final.wav")
        if os.path.exists(fallback_path):
            job = {"status": "done", "file": fallback_path}
        else:
            return jsonify({"error": "Not ready"}), 404

    preview = request.args.get("preview")
    if preview:
        return send_file(job["file"], mimetype="audio/wav", as_attachment=False)

    filename = request.args.get("filename", "conversation.wav").strip()
    if not filename.endswith(".wav"):
        filename += ".wav"

    # Save a named copy to audio_samples/
    dest_path = os.path.join(AUDIO_SAMPLES_DIR, filename)
    shutil.copy2(job["file"], dest_path)

    # Send the saved copy (not the temp file) so the filename is correct
    return send_file(dest_path, mimetype="audio/wav", as_attachment=True,
                     download_name=filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)