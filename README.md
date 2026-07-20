# VoiceCraft — Multilingual Voice Studio 🎙️

An interactive voice synthesis studio interface to create natural dialogues using multi-speaker voice profiles, blending **English** with **20+ Indic languages** in a single seamless audio exchange.

---

## ✨ Features

- **Premium Studio UI**: Frosted glassmorphism card layout, deep warm cocoa theme, glowing background ambient orbs, and gold highlights.
- **Waveform Visualizer**: Live animated vertical-bar audio frequency spectrum graph.
- **Active Cast Selection**: Dynamic interactive speaker cards detailing characteristics (e.g. *Warm, measured*, *Soft, natural*) with active highlight outlines.
- **Script Builder**: Add and delete dialogues turn-by-turn or drop scripts into the **Quick Script** parser for bulk dialogue generation (using `Speaker [lang]: dialogue` syntax).
- **Auto-Detection**: Advanced Unicode character range analysis auto-detects language scripts (Gujarati, Punjabi, Bengali, Telugu, Tamil, Kannada, Malayalam, Odia, Urdu, Hindi, Sanskrit, etc.).
- **Live Status Feed**: Progress bar with a glowing gradient animation, live spin loader, and active model/language tags during synthesis.
- **Custom Downloads**: Export WAV files with personalized names directly from the UI.

---

## 🛠️ Architecture & Backends

- **English Synthesis**: Coqui VITS (`tts_models/en/vctk/vits`)
- **Indic Synthesis**: Indic Parler TTS (`ai4bharat/indic-parler-tts`)
- **Server Stack**: Flask, PyTorch, Hugging Face Transformers, soundfile, pydub

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.9 – 3.11** (Python 3.10 is highly recommended; Python 3.12+ is not supported by Coqui TTS).
- **espeak-ng**: System-level phonemizer backend required for English VITS.

### 1. Install `espeak-ng`

#### Windows (Command Prompt / PowerShell):
```powershell
winget install --id eSpeak-NG.eSpeak-NG --accept-source-agreements --accept-package-agreements
```
*Note: Ensure `C:\Program Files\eSpeak NG` is added to your environment `PATH` variable.*

#### Linux (Debian/Ubuntu):
```bash
sudo apt-get update
sudo apt-get install espeak-ng -y
```

### 2. Clone the Repository
```bash
git clone https://github.com/reh-31/tts_multilingual.git
cd tts_multilingual
```

### 3. Install Dependencies
```bash
pip install flask TTS pydub parler-tts transformers torch soundfile
```

---

## 🏃 Running the Studio

Start the Flask application server:
```bash
python tts_app.py
```

Access the studio dashboard at:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📝 Script Format Example
You can load scripts into the **Quick Script** field using the format below:
```text
Doctor [en]: Good morning. What brings you here today?
Patient [hi]: मुझे दो दिनों से पेट में दर्द हो रहा है।
Doctor [mr]: तुम्हाला ताप किंवा उलट्या होत आहेत का?
Patient [gu]: ના, પણ મને ઉબકા આવે છે.
```
*Note: Script parser dynamically maps speaker names to active cast profiles.*