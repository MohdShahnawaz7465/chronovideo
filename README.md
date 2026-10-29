# ChronoVideo ⚡
### Time-Indexed Multimodal Video Intelligence & Semantic RAG Platform

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-fc521f.svg?style=flat)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/AI_Engine-Gemini_3.8_Flash-4285F4.svg?style=flat&logo=google)](https://ai.google.dev)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vercel Live](https://img.shields.io/badge/Live_Demo-chronovideo--3u7t.vercel.app-000000.svg?style=flat&logo=vercel)](https://chronovideo-3u7t.vercel.app)

> 🌐 **Live Web Application:** [https://chronovideo-3u7t.vercel.app](https://chronovideo-3u7t.vercel.app)  
> 
> **Solving the Technical Video Consumption Bottleneck:**  
> Students, researchers, and engineers spend hours manually scrubbing through 2-to-3-hour lectures searching for specific derivations, algorithmic proofs, or code walkthroughs.  
> **ChronoVideo** ingests high-density lectures, performs temporal sliding-window vectorization, detects semantic topic change-points via unsupervised cosine distance clustering, and indexes chunks into a ChromaDB vector store. Users query in natural language to receive instant, synthesised AI answers with **interactive, zero-latency clickable timestamp anchors** (`[▶ 04:49]`) that jump the video to the exact second.

---

## 🚀 Key Highlights & Capabilities

- 🎯 **Sub-Second Semantic Moment Retrieval:**  
  Ask conceptual questions (e.g. *"Where is the chain rule applied in backpropagation?"* or *"Explain how the learning rate controls step size"*). ChronoVideo retrieves the top-$K$ semantic chunks and provides timestamped citations that seek the embedded YouTube/HTML5 player in real time.
- 🧠 **Unsupervised Machine Learning Topic Clustering:**  
  Formulates lecture segmentation as a sequential vector change-point detection problem. Uses cosine distance gradient spikes across continuous latent embeddings to automatically divide uninterrupted lectures into coherent chapters.
- ⚡ **2-Stage Hybrid RAG Architecture:**  
  Combines dense vector retrieval over HNSW indexes (ChromaDB) with lexical keyword matching, synthesized via Google Gemini 3.8 Flash (with resilient algorithmic heuristics fallback for offline environments).
- 📜 **Live Synchronized Transcript Drawer:**  
  Interactive, synchronized transcript stream with active sentence glowing as the lecture progresses, complete with dual-action jump triggers (in-player jump or external deep-link).
- 📝 **Authentic Ruled Notebook & 1-Click PDF Export:**  
  Generates real student-style ruled notebook study notes with red margin lines, blue ruling, timestamp badges, formula callout boxes, and print-ready vector PDF downloads (`POST /api/export-notes-pdf`).
- 🌐 **Multi-Language Lecture Notes Engine:**  
  Instant AI translation into 10 languages (English default, Hinglish, हिन्दी, Español, Français, Deutsch, 日本語, বাংলা, தமிழ், తెలుగు) preserving code, LaTeX equations, and timestamp anchors.
- 📁 **Offline File Support (PDF Slides & Media):**  
  Drop professor slides (`.pdf`) or recorded lecture audio/video (`.mp4`, `.mp3`, `.mkv`, `.webm`) processed locally with Groq Whisper large-v3-turbo and PyMuPDF.
- 📱 **100% Mobile Responsive & Zero-Overflow UI:**  
  Tested and audited across mobile (360px–480px), tablet (768px), and 4K desktop displays with fluid typography and dark mode aesthetic.

---

## 🛠️ System Architecture

```
                      [ YouTube URL / Technical Lecture ]
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
              [ Spoken Stream ]                 [ Temporal Indexing ]
        (YouTube Transcript Engine)        (Sliding Window Chunking 35-45s)
                      │                                 │
                      └────────────────┬────────────────┘
                                       ▼
                       [ Temporal Embedding Generation ]
                          (384-dim Dense Vector Space)
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
      [ ChromaDB Persistent Vector DB ]    [ Unsupervised ML Clustering ]
        - HNSW Cosine Indexing               - Cosine Drift Change-Point Detection
        - Metadata: start_sec, end_sec       - Automated Chapter Title Synthesis
                      │                                 │
                      │ ◄────── Semantic Query ─────────┤
                      ▼
          [ Top-K Context Retrieval ]
                      │
                      ▼
         [ Gemini 3.8 Flash RAG Engine ]
                      │
                      ▼
   [ Interactive Workspace UI: Linear Obsidian Dark Mode ]
   ├── Clickable Timestamp Anchors [ ▶ Jump in Player (04:49) ]
   ├── Synchronized Transcript Drawer with Auto-Scroll
   └── KaTeX Mathematical Study Sheet Modal
```

---

## 📚 Machine Learning Formulation

### 1. Unsupervised Temporal Chapter Boundary Detection
Let $V = [\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_T]$ represent sequential $d$-dimensional sentence embedding vectors for consecutive temporal windows $t \in [1, T]$ of duration $\Delta t \approx 40\text{s}$.

The conceptual drift between consecutive temporal segments is modeled by the Cosine Distance:
$$\mathcal{D}(t, t+1) = 1 - \frac{\mathbf{v}_t \cdot \mathbf{v}_{t+1}}{\|\mathbf{v}_t\|_2 \|\mathbf{v}_{t+1}\|_2}$$

A lecture chapter transition $\tau^*$ is registered whenever the temporal rate of change exceeds an adaptive moving threshold:
$$\tau^* = \{ t \mid \mathcal{D}(t, t+1) > \mu_{\mathcal{D}} + \alpha \cdot \sigma_{\mathcal{D}} \}$$
where $\mu_{\mathcal{D}}$ and $\sigma_{\mathcal{D}}$ are the mean and standard deviation of conceptual drift across the lecture, and $\alpha$ is a sensitivity hyperparameter ($0.4 \le \alpha \le 0.8$).

### 2. Hybrid RAG Retrieval Score
Given a user query $q$, the ranking score for candidate temporal chunk $c_i$ balances dense semantic similarity and term specificity:
$$\text{Score}(q, c_i) = \lambda \cdot \cos(\mathbf{e}_q, \mathbf{e}_{c_i}) + (1 - \lambda) \cdot \text{BM25}(q, c_i)$$

---

## 📂 Project Structure

```
chronovideo/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── router.py          # FastAPI Endpoints (/api/index, /api/search, /api/generate-notes)
│   │   ├── services/
│   │   │   ├── ingestion.py       # YouTube Transcript & Temporal Window Chunking
│   │   │   ├── ml_clustering.py   # Unsupervised Cosine Change-Point Clustering
│   │   │   ├── vector_db.py       # ChromaDB Persistent Storage & HNSW Indexing
│   │   │   └── rag_synthesizer.py # Gemini 3.8 Flash RAG & LaTeX Note Generator
│   │   ├── config.py              # Centralized Settings & Environment Paths
│   │   └── main.py                # FastAPI ASGI App & Static File Mounts
│   ├── data/
│   │   └── chroma_storage/        # Persistent Local Vector Store (Git-Ignored)
│   ├── .env.example               # Example Configuration File
│   └── .env                       # Local Secrets (Ignored)
├── frontend/
│   └── index.html                 # Linear-Style Dark Obsidian Single-Page Application
├── requirements.txt               # Production Python Dependencies
├── run.py                         # Single-Command Application Launcher
└── README.md                      # Comprehensive Architecture & Interview Documentation
```

---

## ⚡ Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/chronovideo.git
cd chronovideo

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)
Copy `.env.example` to `backend/.env`:
```bash
cp backend/.env.example backend/.env
```
Add your Google Gemini API key to `backend/.env` for optimal synthesis (the system automatically falls back to deterministic extraction if no key is provided).

### 4. Launch the Platform
```bash
python run.py
```
Open your browser at:
- 🌐 **Production Web Platform (Vercel):** [https://chronovideo-3u7t.vercel.app](https://chronovideo-3u7t.vercel.app)
- **Local Dev Server:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🚢 Deployment Guide (GitHub, Vercel & Cloud)

### 1. Push Code to GitHub
```bash
# Initialize and push to your new GitHub repository
git remote add origin https://github.com/<your-username>/chronovideo.git
git branch -M main
git push -u origin main
```

### 2. Deploy Full-Stack to Render / Railway / Fly.io (Recommended)
Because ChronoVideo uses FastAPI, ChromaDB, and speech models, it runs as a persistent service:
1. Connect your GitHub repository to **[Render](https://render.com)** or **[Railway](https://railway.app)**.
2. Select **Python Web Service**.
3. **Build Command:** `pip install -r requirements.txt`
4. **Start Command:** `python run.py`
5. Under Environment Variables, add:
   - `GROQ_API_KEY`: *(your groq key for ultra-fast Whisper & Qwen translation)*
   - `PORT`: `8000` (or let Render set it automatically)

### 3. Deploy Frontend on Vercel
1. Import repository on **[Vercel](https://vercel.com)**.
2. The included `vercel.json` automatically routes static traffic to `/frontend/index.html`.
3. To proxy `/api/*` requests to your hosted backend, add your backend URL in `vercel.json` rewrites.

---

## 💼 Resume Bullets (For Day-1 Placements)

> **ChronoVideo | Time-Indexed Multimodal Video Intelligence & Semantic RAG Platform**  
> *Tech Stack: Python, FastAPI, ChromaDB, Gemini 3.8 Flash, YouTube API, NumPy, HTML5/CSS3*
>
> - Architected a high-throughput video intelligence platform that converts multi-hour technical lectures into time-indexed semantic vector spaces with **sub-second retrieval latency**.
> - Formulated an **unsupervised temporal clustering algorithm** using cosine distance change-point detection across sequential sentence embeddings to automatically partition uninterrupted lectures into structured chapters.
> - Implemented a **2-Stage Hybrid RAG pipeline** using ChromaDB HNSW indexing and Gemini 3.8 Flash to synthesize natural language explanations paired with **actionable timestamp anchors** (`[▶ MM:SS]`).
> - Engineered an interactive dark obsidian web interface with an embedded YouTube IFrame API integration, featuring programmatic dual-action playback seek (`player.seekTo()`) and live transcript synchronization.

---

## 🎓 Viva & Technical Interview Cheat-Sheet

| Interview Question | How to Answer |
| :--- | :--- |
| **Why use temporal sliding windows instead of single sentences?** | Single sentences lack contextual semantics (e.g. *"This means that..."* has zero meaning alone). A 35–45 second sliding window encapsulates complete mathematical steps or algorithmic ideas while maintaining pinpoint timestamp precision. |
| **How does unsupervised clustering detect chapter changes?** | Adjacent temporal windows within the same topic share high cosine similarity ($\ge 0.85$). When an instructor pivots (e.g., from Gradient Descent theory to Code Implementation), the cosine distance between consecutive vectors spikes, indicating a change-point boundary. |
| **Why ChromaDB over a traditional relational database?** | Traditional databases cannot compute nearest-neighbor distance metrics across high-dimensional latent vectors. ChromaDB provides persistent HNSW (Hierarchical Navigable Small World) graph indexing with $O(\log N)$ query complexity. |
| **What happens if the Gemini API rate limit is exceeded?** | The platform incorporates an intelligent fallback mechanism: it ranks top-$K$ chunks via vector similarity and extracts high-confidence timestamped sentence boundaries algorithmically, ensuring 100% uptime. |

---

## 📄 License
Released under the [MIT License](LICENSE).
