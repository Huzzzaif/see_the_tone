# See-the-Tone Plus — Voice Activity Detection Module

This repository is the foundation for an **open-source speech emotion analysis** project.  
The first milestone implemented here is **Voice Activity Detection (VAD)** — the step that strips silence and noise so later AI models can focus only on speech.

---

## 📌 What’s Done So Far

### 1. Project Setup
- Created a clean folder structure:
see-the-tone-plus/
core/ # Core signal processing & ML modules
train/ # Training scripts & utilities
app/ # UI / API (later)
- Installed core dependencies:
- `webrtcvad` for speech detection
- `librosa` + `soundfile` for audio loading/decoding (supports `.wav`, `.mp3`, `.m4a`, etc.)
- `numpy` for numeric processing

---

### 2. VAD Implementation
- **Function:** `detect_speech_intervals(wav_path, ...)`
- **What it does:**
1. Loads audio from file (any format `ffmpeg` supports, e.g. `.m4a`)
2. Resamples to **16 kHz mono** (VAD requirement)
3. Converts to **PCM16 bytes**
4. Splits into **exact 10/20/30 ms frames**
5. Uses **WebRTC VAD** to label each frame as speech/no-speech
6. Smooths start/stop decisions using a hysteresis ring buffer
7. Collapses results into `(start_sec, end_sec)` speech intervals
8. Drops very short segments, merges close ones

- **Why:** Clean speech intervals are critical for reliable feature extraction (pitch, energy, embeddings) and efficient model inference.

---

### 3. CLI Tool for Testing
- `train/prepare_data.py`
- Usage:
```bash
python train/prepare_data.py "Paseo Vera.m4a" --aggr 2 --frame_ms 30 --padding_ms 300
