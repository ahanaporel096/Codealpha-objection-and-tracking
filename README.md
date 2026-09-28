# VisionTrack AI — Real-Time Object Detection & Tracking

A multi-runtime computer vision project supporting both **Local Streamlit Mode (YOLOv8 + PyTorch)** and **Cloud Vercel Web Mode (Browser TensorFlow.js + Serverless API)**.

---

## 🚀 Choose Your Mode

### Mode 1: Local Streamlit Mode (YOLOv8 + PyTorch)
Runs locally on your computer with desktop hardware camera acceleration, full Ultralytics YOLOv8 models (`yolov8n.pt`), and Python OpenCV.

#### How to Run:
```bash
# 1. Navigate to python_app
cd python_app

# 2. Activate virtual environment
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On Linux/macOS

# 3. Launch Streamlit
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

### Mode 2: Vercel Web Mode (Browser Vision + Serverless)
A zero-install, serverless-ready web app hosted directly on Vercel. 
* Uses **HTML5 `navigator.mediaDevices.getUserMedia()`** for in-browser webcam capture.
* Runs **COCO-SSD (MobileNet v2)** directly in the browser via WebGL GPU acceleration.
* Implements an **IoU Object Tracker** for persistent ID tracking across frames.
* Connects to **Groq AI** for automated scene intelligence reports via `/api/ai_summary`.
* Syncs detection logs to **Supabase** via `/api/sync_supabase`.

#### Deploy to Vercel:
1. Push your repository to GitHub.
2. Import the project in your [Vercel Dashboard](https://vercel.com/new).
3. Under **Project Settings ➔ Environment Variables**, add:

| Variable | Description | Example |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | Your Groq API Key | `gsk_...` |
| `GROQ_MODEL` | (Optional) Active Groq Model | `openai/gpt-oss-20b` |
| `SUPABASE_URL` | Your Supabase Project URL | `https://xxxx.supabase.co` |
| `SUPABASE_KEY` | Your Supabase Anon/Service Key | `eyJhbGciOi...` |

4. Click **Deploy**. Vercel will serve `index.html` at your production URL and route `/api/*` to the serverless Python functions automatically.

#### How to Run Vercel Mode Locally:
You can test the static web interface locally with any static web server:
```bash
npx serve .
# or
python -m http.server 3000
```
Open **`http://localhost:3000`** in your browser.

---

## 📁 Project Architecture

```
├── index.html                 ← Vercel web frontend entrypoint
├── styles.css                 ← Responsive styles for desktop & mobile
├── app.js                     ← Browser detection engine, IoU tracker & UI logic
├── vercel.json                ← Vercel deployment routes and serverless config
│
├── api/                       ← Lightweight Vercel Serverless Functions
│   ├── ai_summary.py          ← Groq AI scene intelligence endpoint
│   └── sync_supabase.py       ← Supabase cloud logging endpoint
│
└── python_app/                ← Untouched Local Streamlit Application
    ├── app.py                 ← Streamlit entry point
    ├── requirements.txt       ← Local Python dependencies
    ├── supabase_schema.sql    ← Database table setup script
    ├── .env                   ← Local secrets & API keys
    └── src/
        ├── detector.py        ← YOLOv8 model wrapper
        ├── tracker.py         ← ByteTrack / BoT-SORT helper
        ├── video_processor.py ← OpenCV video capture
        ├── utils.py           ← Drawing, HUD, CSV & snapshot exports
        ├── ai_assistant.py    ← Local Groq AI client
        └── supabase_client.py ← Local Supabase client
```

---

## 🔒 Security & Privacy
* Secret API keys are **never bundled in client-side JavaScript**.
* In Vercel mode, browser calls route through serverless endpoints (`/api/ai_summary`, `/api/sync_supabase`) which read environment variables securely on the server.
* Local `.env` files are excluded by `.gitignore`.
