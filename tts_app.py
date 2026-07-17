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
  <title>VoiceCraft — Multilingual Dialogue</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0c0c10; --surface: #13131a; --surface2: #1c1c26;
      --border: #2a2a38; --accent: #7c6af7; --accent2: #f0b429;
      --text: #e8e8f0; --muted: #6b6b80;
      --success: #3ecf8e; --error: #f87171;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Mono', monospace;
      background: var(--bg); color: var(--text);
      min-height: 100vh; overflow-x: hidden;
    }
    body::before {
      content: ''; position: fixed; inset: 0;
      background:
        radial-gradient(ellipse 60% 50% at 20% 10%, rgba(124,106,247,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 80% 80%, rgba(240,180,41,0.07) 0%, transparent 60%);
      pointer-events: none; z-index: 0;
    }
    .wrapper { position: relative; z-index: 1; max-width: 920px; margin: 0 auto; padding: 48px 24px 80px; }
    header { margin-bottom: 48px; }
    .logo-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
    .logo-icon { width: 36px; height: 36px; background: var(--accent); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
    h1 {
      font-family: 'Syne', sans-serif; font-size: clamp(28px,5vw,42px);
      font-weight: 800; letter-spacing: -1px;
      background: linear-gradient(135deg, var(--text) 40%, var(--accent));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: var(--muted); font-size: 13px; letter-spacing: 0.02em; }
    .lang-badge {
      display: inline-flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;
    }
    .lang-badge span {
      background: var(--surface2); border: 1px solid var(--border);
      border-radius: 6px; padding: 3px 10px; font-size: 11px; color: var(--muted);
    }
    .section-label {
      font-family: 'Syne', sans-serif; font-size: 11px; font-weight: 700;
      letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 12px;
    }
    .speakers-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 32px; }
    .speaker-chip {
      display: flex; align-items: center; gap: 8px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 8px 14px; transition: border-color 0.2s;
    }
    .speaker-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .speaker-name { font-size: 13px; font-weight: 500; text-transform: capitalize; }
    #dialogue-list { display: flex; flex-direction: column; gap: 10px; margin-bottom: 16px; }

    /* 4-column row: speaker | language | text | delete */
    .dialogue-row {
      display: grid;
      grid-template-columns: 130px 120px 1fr 44px;
      gap: 8px; align-items: stretch; animation: slideIn 0.25s ease;
    }
    @keyframes slideIn { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:translateY(0); } }
    select, textarea {
      background: var(--surface); border: 1px solid var(--border);
      color: var(--text); font-family: 'DM Mono', monospace; font-size: 13px;
      border-radius: 10px; padding: 10px 14px; outline: none; transition: border-color 0.2s; width: 100%;
    }
    select:focus, textarea:focus { border-color: var(--accent); }
    select {
      cursor: pointer; appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b6b80' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 10px center; padding-right: 28px;
    }
    /* Colour-coded language selects */
    select.lang-en { border-color: #4a8cff44; }
    select.lang-hi { border-color: #ff9933aa; }
    select.lang-mr { border-color: #3ecf8eaa; }
    select.lang-gu { border-color: #f0b42988; }
    textarea { resize: none; height: 48px; line-height: 1.5; }
    .bulk-textarea { height: 140px; margin-bottom: 10px; }
    .delete-btn {
      background: transparent; border: 1px solid var(--border); color: var(--muted);
      cursor: pointer; border-radius: 10px; width: 44px; font-size: 16px;
      transition: all 0.2s; display: flex; align-items: center; justify-content: center;
    }
    .delete-btn:hover { border-color: var(--error); color: var(--error); background: rgba(248,113,113,0.08); }
    .action-bar { display: flex; gap: 10px; margin-bottom: 32px; flex-wrap: wrap; }
    .btn {
      font-family: 'Syne', sans-serif; font-weight: 700; font-size: 13px;
      letter-spacing: 0.05em; border: none; border-radius: 10px;
      padding: 12px 22px; cursor: pointer; transition: all 0.2s;
      display: flex; align-items: center; gap: 8px;
    }
    .btn-ghost { background: var(--surface); color: var(--text); border: 1px solid var(--border); }
    .btn-ghost:hover { border-color: var(--accent); color: var(--accent); }
    .btn-primary { background: var(--accent); color: #fff; box-shadow: 0 4px 20px rgba(124,106,247,0.35); }
    .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 6px 28px rgba(124,106,247,0.45); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-success { background: var(--success); color: #000; }
    .btn-success:hover { filter: brightness(1.1); }
    #status-box { display:none; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 24px; margin-bottom: 24px; }
    .progress-label { font-size: 12px; color: var(--muted); margin-bottom: 10px; display: flex; justify-content: space-between; }
    .progress-bar-bg { background: var(--surface2); border-radius: 99px; height: 6px; overflow: hidden; }
    .progress-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.4s ease; width: 0%; }
    .status-msg { margin-top: 14px; font-size: 13px; color: var(--muted); }
    .model-info { display:none; margin-top: 10px; padding: 8px 14px; border-radius: 8px; background: var(--surface2); border: 1px solid var(--border); font-size: 12px; display: flex; gap: 10px; align-items: center; }
    .mi-lang-badge { font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; }
    .mi-lang-en { background: rgba(96,165,250,.15); color: #60a5fa; border: 1px solid rgba(96,165,250,.3); }
    .mi-lang-indic { background: rgba(74,222,128,.15); color: #4ade80; border: 1px solid rgba(74,222,128,.3); }
    .mi-model-name { color: var(--text); font-family: 'DM Mono', monospace; }
    #audio-section { display:none; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 24px; }
    .audio-title { font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .audio-title::before { content:''; display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--success); animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:0.5;transform:scale(1.4);} }
    audio { width:100%; margin-bottom:16px; border-radius:8px; accent-color: var(--accent); }
    .transcript { border-top: 1px solid var(--border); padding-top: 16px; margin-top: 4px; display:flex; flex-direction:column; gap:8px; }
    .transcript-line { display:flex; gap:10px; font-size:12px; align-items:baseline; }
    .t-role { font-family:'Syne',sans-serif; font-weight:700; min-width:80px; text-transform:capitalize; font-size:11px; letter-spacing:0.05em; }
    .t-lang { font-size:10px; color: var(--muted); min-width:24px; }
    .t-text { color: var(--muted); }
    .error-msg { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: var(--error); border-radius:10px; padding:12px 16px; font-size:13px; margin-top:16px; display:none; }
    .col-header { font-size:10px; color:var(--muted); padding: 0 4px 4px; letter-spacing:0.08em; text-transform:uppercase; }
    .col-headers { display:grid; grid-template-columns: 130px 120px 1fr 44px; gap:8px; margin-bottom:4px; }
  </style>
</head>
<body>
<div class="wrapper">
  <header>
    <div class="logo-row">
      <div class="logo-icon">🎙</div>
      <h1>VoiceCraft</h1>
    </div>
    <p class="subtitle">// multilingual dialogue · Hindi · Marathi · Gujarati · English</p>
    <div class="lang-badge">
      <span>🇮🇳 Hindi हिंदी</span>
      <span>🇮🇳 Marathi मराठी</span>
      <span>🇮🇳 Gujarati ગુજરાતી</span>
      <span>🇬🇧 English</span>
    </div>
  </header>

  <div class="section-label">Speakers</div>
  <div class="speakers-grid" id="speakers-grid"></div>

  <div class="section-label">Bulk Input</div>
  <textarea id="bulk-input" class="bulk-textarea" placeholder="Doctor [hi]: नमस्ते, आप कैसे हैं?
Patient [en]: I have a headache.
Doctor [mr]: तुम्हाला ताप आहे का?
Patient [gu]: ના, ફક્ત ઉબકા."></textarea>
  <div class="action-bar" style="margin-bottom: 24px;">
    <button class="btn btn-ghost" onclick="convertBulkToDialogue()">↳ Convert to Dialogue</button>
  </div>

  <div class="section-label">Dialogue</div>
  <div class="col-headers">
    <span class="col-header">Speaker</span>
    <span class="col-header">Language</span>
    <span class="col-header">Text</span>
    <span class="col-header"></span>
  </div>
  <div id="dialogue-list"></div>

  <div class="action-bar">
    <button class="btn btn-ghost" onclick="addRow()">＋ Add Line</button>
    <button class="btn btn-ghost" onclick="loadExample()">📋 Load Example</button>
    <button class="btn btn-primary" id="gen-btn" onclick="generateAudio()">▶ Generate Audio</button>
  </div>

  <div id="error-msg" class="error-msg"></div>

  <div id="status-box">
    <div class="progress-label">
      <span>Synthesizing voices...</span>
      <span id="progress-count">0 / 0</span>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar-fill" id="progress-fill"></div></div>
    <div class="status-msg" id="status-msg">Starting...</div>
    <div class="model-info" id="model-info">
      <span>Detected:</span>
      <span class="mi-lang-badge" id="mi-lang">—</span>
      <span>→</span>
      <span class="mi-model-name" id="mi-model">—</span>
    </div>
  </div>

  <div id="audio-section">
    <div class="audio-title">Audio ready</div>
    <audio id="audio-player" controls></audio>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <input type="text" id="download-name" placeholder="filename" value="conversation"
        style="background:var(--surface2);border:1px solid var(--border);color:var(--text);
               font-family:'DM Mono',monospace;font-size:13px;border-radius:10px;
               padding:10px 14px;outline:none;width:200px;" />
      <span style="color:var(--muted);font-size:13px;">.wav</span>
      <button class="btn btn-success" onclick="downloadAudio()">⬇ Download WAV</button>
      <button class="btn btn-ghost" onclick="resetAll()">↺ New Dialogue</button>
    </div>
    <div class="transcript" id="transcript-preview"></div>
  </div>
</div>

<script>
  const SPEAKER_NAMES = __SPEAKER_NAMES__;
  const SPEAKER_COLORS = ['#7c6af7','#f0b429','#3ecf8e','#f87171','#60a5fa'];

  const LANGUAGES = [
    { code: 'en', label: 'English 🇬🇧' },
    { code: 'hi', label: 'Hindi हिंदी' },
    { code: 'mr', label: 'Marathi मराठी' },
    { code: 'gu', label: 'Gujarati ગુજ.' },
  ];
  const LANG_FLAGS = { en: '🇬🇧', hi: '🇮🇳hi', mr: '🇮🇳mr', gu: '🇮🇳gu' };

  // Bulk input shorthand: [en] [hi] [mr] [gu]
  const BULK_LANG_MAP = { en:'en', hi:'hi', mr:'mr', gu:'gu', hindi:'hi', marathi:'mr', gujarati:'gu', english:'en' };

  let rows = [];
  let rowCounter = 0;
  let pollingTimer = null;
  let currentJobId = null;

  function renderSpeakers() {
    const grid = document.getElementById('speakers-grid');
    grid.innerHTML = '';
    SPEAKER_NAMES.forEach((name, i) => {
      const chip = document.createElement('div');
      chip.className = 'speaker-chip';
      chip.innerHTML = `
        <span class="speaker-dot" style="background:${SPEAKER_COLORS[i % SPEAKER_COLORS.length]}"></span>
        <span class="speaker-name">${name.replace(/_/g,' ')}</span>`;
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
    const spkOpts = SPEAKER_NAMES.map(n =>
      `<option value="${n}" ${n === role ? 'selected' : ''}>${n.replace(/_/g,' ')}</option>`
    ).join('');
    div.innerHTML = `
      <select id="role-${id}">${spkOpts}</select>
      ${buildLangSelect(lang, id)}
      <textarea id="text-${id}" placeholder="Type dialogue here...">${text}</textarea>
      <button class="delete-btn" onclick="removeRow('${id}')">✕</button>`;
    list.appendChild(div);
  }

  function removeRow(id) {
    document.getElementById(id)?.remove();
    rows = rows.filter(r => r !== id);
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

  // ── Speaker name aliases (Hindi / Marathi / Gujarati → canonical English) ──
  const HINDI_SPEAKER_ALIASES = {
    // English
    'doctor': 'doctor', 'patient': 'patient', 'narrator': 'narrator',
    'custom_1': 'custom_1', 'custom_2': 'custom_2',
    // Hindi
    'डॉक्टर': 'doctor', 'मरीज': 'patient', 'मरीज़': 'patient', 'वर्णनकर्ता': 'narrator',
    // Marathi
    'डॉक्टर': 'doctor', 'रुग्ण': 'patient', 'निवेदक': 'narrator',
    // Gujarati
    'ડૉક્ટર': 'doctor', 'દર્દી': 'patient', 'નિવેદક': 'narrator',
  };

  // Detect language from script when no [lang] tag is provided.
  // Gujarati (U+0A80-U+0AFF) is detected first as it has its own script.
  // Marathi and Hindi both use Devanagari — must use explicit [mr] tag to get Marathi.
  function _detectLang(text, prefix) {
    const all = text + ' ' + prefix;
    const codes = [...all].map(c => c.charCodeAt(0));
    if (codes.some(c => c >= 0x0A80 && c <= 0x0AFF)) return 'gu';  // Gujarati script
    if (codes.some(c => c >= 0x0900 && c <= 0x097F)) return 'hi';  // Devanagari → default Hindi
    return 'en';
  }

  // ── Bulk parser: supports "Speaker [lang]: text" and Hindi "डॉक्टर: text" ─
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

      // Extract optional [lang] tag: "Doctor [hi]" or "डॉक्टर [hi]"
      const langMatch = prefix.match(/\\[([a-z]+)\\]\\s*$/i);
      let langRaw = langMatch ? langMatch[1].toLowerCase() : null;
      const speakerRaw = prefix.replace(/\\[[^\\]]+\\]/, '').trim().toLowerCase().replace(/\\s+/g, '_');

      // Resolve speaker: strip [lang] tag, then try alias map, then exact lookup, then alternate fallback
      const prefixClean = prefix.replace(/\\[[^\\]]+\\]/, '').trim();
      const speaker = HINDI_SPEAKER_ALIASES[prefixClean]
                   || HINDI_SPEAKER_ALIASES[prefixClean.toLowerCase()]
                   || speakerLookup.get(speakerRaw)
                   || (added % 2 === 0 ? 'doctor' : 'patient');  // fallback: alternate by line

      // Auto-detect language from script if no [lang] tag
      if (!langRaw) {
        langRaw = _detectLang(text, prefix);
      }
      const lang = BULK_LANG_MAP[langRaw] || 'en';

      addRow(speaker, text, lang);
      added++;
    }

    if (added === 0) {
      showError('No valid lines found. Make sure each line has format: Speaker: text  (colon after speaker name). For Gujarati: ડૉક્ટર: ટેક્સ્ટ');
      return;
    }
    hideError();
  }

  // ── Generate ──────────────────────────────────────────────────────────────
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
      const color = SPEAKER_COLORS[SPEAKER_NAMES.indexOf(role) % SPEAKER_COLORS.length] || '#aaa';
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