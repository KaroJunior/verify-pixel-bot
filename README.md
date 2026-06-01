# Verify Pixel Bot

An AI-powered Telegram bot that analyzes images and estimates the likelihood that they were generated or significantly modified using AI techniques.

The bot uses classical image forensics methods rather than external APIs or paid AI services.

---

## ⚠️ Disclaimer

This project is experimental and should not be treated as a definitive AI detection system.

Results are probabilistic and may be incorrect, especially with:

* Newer AI models (e.g., FLUX, Midjourney v6+)
* Heavily edited real images
* Low-quality or compressed images

This tool is intended for educational and research purposes only.

---

## 🎯 Features

* Telegram bot interface (built with aiogram)
* Fully local image processing (no external APIs)
* No OpenCV dependency (lightweight stack)
* Privacy-focused (no image storage)
* Automatic temporary file cleanup
* Multi-layer analysis system:

### 1. Metadata Analysis

* EXIF data extraction
* Camera information detection
* AI/editing software signature detection

### 2. Frequency Domain Analysis

* Fast Fourier Transform (FFT)
* Detection of unnatural frequency patterns
* Compression and smoothing artifacts analysis

### 3. Visual Heuristics

* Texture consistency analysis
* Edge distribution evaluation
* Noise pattern inspection
* Color distribution analysis
* JPEG artifact detection (8x8 block patterns)

---

## 🧠 Scoring System

Final score is calculated using weighted analysis:

```
Final Score =
(0.25 × Metadata Score) +
(0.40 × Frequency Score) +
(0.35 × Heuristic Score)
```

### Interpretation

* 70–100% → Likely AI-generated
* 40–69% → Possibly AI-generated
* 20–39% → Likely authentic
* 0–19% → Very likely authentic

Confidence is derived from agreement between all analysis modules.

---

## 💻 Tech Stack

* Python 3.11+
* aiogram (Telegram Bot API)
* Pillow (image processing)
* NumPy (numerical computation)
* SciPy (FFT + signal processing)
* piexif (EXIF metadata handling)
* python-dotenv (environment configuration)

---

## 🚀 Installation

### 1. Clone repository

```bash
git clone <your-repo-url>
cd verify-pixel-bot
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Add your bot token:

```
BOT_TOKEN=your_telegram_bot_token
```

### 5. Create temp directory

```bash
mkdir temp
```

### 6. Run bot

```bash
python bot.py
```

---

## 📁 Project Structure

```
verify-pixel-bot/
├── bot.py
├── analyzer.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── utils/
│   ├── metadata.py
│   ├── frequency.py
│   ├── heuristics.py
│   ├── scoring.py
├── temp/
└── README.md
```

---

## 📊 Performance

* RAM usage: ~100–200MB
* CPU usage: Moderate (FFT-heavy operations)
* Image processing time:

  * Small images: 3–5s
  * Medium images: 5–8s
  * Large images: 8–12s

---

## 🔒 Privacy

* No external API calls
* No image storage (temporary only)
* No logging of image content
* All processing happens locally

---

## ⚙️ Limitations

* Not reliable against latest AI generators
* Can produce false positives/negatives
* Heuristic-based system (not a trained ML model)
* Performance depends on image quality and compression

---

## 🛣️ Future Improvements

* ML-based detector integration (CLIP/SigLIP models)
* GPU acceleration for FFT processing
* Batch image analysis
* Improved scoring calibration
* Web dashboard interface

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built as an experimental AI image forensics tool using Python image processing libraries.
