<div align="center">

<img src="logo.png" alt="DocChat AI Logo" width="120" style="border-radius: 50%"/>

# ⚡ DocChat AI

### Fine-Tuned LLM for Document Q&A & Toxic Comment Detection

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-TinyLlama-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/TinyLlama)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A production-grade, fully local AI system — no cloud APIs, no RAG, no vector databases.**
**Pure fine-tuning. Pure intelligence.**

[🚀 Live Demo](https://gurumsd-docchat-ai.hf.space) · [📖 API Docs](http://localhost:8000/docs) · [🐛 Report Bug](https://github.com/guru22cseb37/DocChat-AI/issues)

</div>

---

## 🌟 What Is DocChat AI?

DocChat AI is an end-to-end intelligent document assistant that runs **100% locally** on your machine. It fine-tunes a small language model (TinyLlama 1.1B) using **LoRA/QLoRA** to understand your documents and detect toxic language — all without sending a single byte to the internet.

Upload your resume, research paper, or any document — then have a real conversation with it. Switch to Toxicity mode and instantly analyze comments with detailed labels and confidence scores.

> **No RAG. No vector databases. No cloud APIs. Just fine-tuned intelligence.**

---

## ✨ Features at a Glance

| Feature | Description |
|---|---|
| 📄 **Document Q&A** | Upload TXT, PDF, or DOCX — ask anything, get precise answers |
| ☣️ **Toxicity Detection** | 6-label classification with confidence scores and explanations |
| 🧠 **QLoRA Fine-Tuning** | TinyLlama fine-tuned with LoRA adapters on Jigsaw + synthetic Q&A data |
| ⚡ **FastAPI Backend** | Async API with auth, batch inference, token truncation, and graceful error handling |
| 🎨 **Premium Streamlit UI** | Glassmorphism design, animated chat bubbles, real-time stats |
| 🛡️ **API Key Auth** | All endpoints secured with `X-API-Key` header |
| 📊 **Evaluation Metrics** | F1, ROUGE-L, and BERT Score — runs without retraining |
| 🌐 **HF Spaces Deployed** | UI live at [gurumsd-docchat-ai.hf.space](https://gurumsd-docchat-ai.hf.space) |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│  ┌──────────────┐         ┌────────────────────┐    │
│  │ Document Q&A │         │  Toxicity Checker   │    │
│  │   Mode       │         │      Mode           │    │
│  └──────┬───────┘         └────────┬───────────┘    │
└─────────┼───────────────────────────┼────────────────┘
          │    HTTP + X-API-Key        │
          ▼                           ▼
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│  POST /ask    POST /predict    POST /batch_predict   │
│  GET  /health                                        │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│           TinyLlama-1.1B + LoRA Adapter             │
│    Trained on: Jigsaw Toxicity + Synthetic Q&A       │
│    Runs on:    CPU (local) | GPU (cloud)             │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
docchat-ai/
│
├── 📄 api.py               # FastAPI backend — /ask, /predict, /batch_predict
├── 🎨 ui.py                # Streamlit chat UI with glassmorphism design
├── 🧠 train.py             # QLoRA fine-tuning script with OOM recovery
├── 📊 evaluate.py          # F1, ROUGE-L, BERT Score evaluation
│
├── data/
│   ├── generate_qa.py      # Synthetic Q&A pair generation
│   └── prepare_toxic.py    # Jigsaw → instruction format converter
│
├── adapter_weights/        # Saved LoRA adapter after training
├── checkpoints/            # Mid-training checkpoints for crash recovery
│
├── logo.png                # App logo
├── background.png          # App background
├── pyproject.toml          # uv-managed dependencies
├── requirements.txt        # For HF Spaces deployment
└── README.md               # You are here 👋
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- 8GB+ RAM (16GB recommended for TinyLlama on CPU)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/guru22cseb37/DocChat-AI.git
cd DocChat-AI

# 2. Install uv (if not already installed)
pip install uv

# 3. Install all dependencies
uv sync
```

### Running the Application

Open **two separate terminals** and run:

**Terminal 1 — Start the FastAPI Backend:**
```bash
uv run uvicorn api:app --reload --port 8000
```

**Terminal 2 — Start the Streamlit UI:**
```bash
uv run streamlit run ui.py
```

Then open your browser at **http://localhost:8501** 🎉

> **Alternative (without uv):**
> ```bash
> pip install -r requirements.txt
> python -m uvicorn api:app --port 8000        # Terminal 1
> python -m streamlit run ui.py                 # Terminal 2
> ```

---

## 🧠 Fine-Tuning the Model

```bash
# Step 1 — Generate synthetic Q&A pairs
uv run python data/generate_qa.py

# Step 2 — Prepare Jigsaw toxicity dataset
uv run python data/prepare_toxic.py

# Step 3 — Run fine-tuning
uv run python train.py
```

> **GPU detected?** → TinyLlama trained with QLoRA (4-bit) — fast and memory-efficient  
> **No GPU (CPU only)?** → Automatically falls back to `sshleifer/tiny-gpt2` — stable and reliable  
> **Training crashes?** → Checkpoints saved every epoch — resumes automatically from last checkpoint

---

## 📡 API Reference

All endpoints require the header: `X-API-Key: docchat-secret-key`

### `GET /health`
Check if the server and model are running.
```json
{ "status": "ok", "model_loaded": true, "adapter_loaded": false }
```

### `POST /ask`
Answer a question from a document.
```json
// Request
{ "doc": "Your document text here...", "question": "What are the main skills?" }

// Response
{ "answer": "The main skills are Python, FastAPI, and machine learning...", "tokens_used": 87, "truncated": false }
```

### `POST /predict`
Classify a comment for toxicity.
```json
// Request
{ "comment": "You are amazing and talented!" }

// Response
{ "label": "safe", "confidence": 0.92, "explanation": "The comment is positive and constructive." }
```

### `POST /batch_predict`
Classify up to 20 comments at once.
```json
// Request
{ "comments": ["Great work!", "I hate you", "This is brilliant"] }
```

---

## 🏷️ Toxicity Labels

| Label | Description |
|---|---|
| `safe` | No toxic content detected |
| `toxic` | Generally offensive content |
| `severe_toxic` | Extremely offensive content |
| `obscene` | Obscene or vulgar language |
| `threat` | Threatening language |
| `insult` | Directed personal insults |
| `identity_hate` | Hate based on identity |

---

## 📊 Evaluation

Run the evaluation script to compute all metrics:

```bash
uv run python evaluate.py
```

**Metrics computed:**
- **F1 Score** — Toxicity classification accuracy (scikit-learn)
- **ROUGE-L** — Answer quality via longest common subsequence
- **BERT Score** — Semantic similarity of generated answers

> ✅ Runs without retraining — loads the saved adapter directly.

---

## ⚙️ Edge Cases Handled

| Scenario | Handling |
|---|---|
| Document exceeds 2000 tokens | Automatically truncated, answered from available content |
| Comment exceeds 512 tokens | Truncated at token boundary — no crash |
| Empty comment submitted | Returns `safe` with "No text to analyse" |
| Non-English input | Returns `unknown` with language warning |
| Server offline | UI shows graceful error message |
| Training OOM crash | Batch size halved, training resumes from checkpoint |
| Incompatible adapter | Falls back to base model — server stays alive |
| Model not loaded | Returns `503` with clear error message |
| Missing request fields | Returns `422` with precise validation message |

---

## 🎨 UI Showcase

| Feature | Preview |
|---|---|
| Document Q&A with logo avatar | Premium glassmorphism chat UI |
| Toxicity result cards | Color-coded SAFE/TOXIC/WARNING cards |
| Neural network background | Futuristic AI-themed design |
| Confidence threshold slider | Filter uncertain predictions in real time |

---

## 🌐 Deployment

**Streamlit UI** is live on Hugging Face Spaces:
> 🔗 **https://gurumsd-docchat-ai.hf.space**

**Note on inference speed:** The model runs on CPU locally (~60 seconds per response). On an NVIDIA GPU or cloud GPU server (AWS g4dn, Google Colab), responses are near-instant (<3 seconds). The architecture is fully production-ready — only the hardware differs.

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-Tuning | PEFT (LoRA/QLoRA) + TRL SFTTrainer |
| Quantization | bitsandbytes (4-bit on GPU) |
| Backend | FastAPI + Uvicorn (async) |
| Frontend | Streamlit (glassmorphism UI) |
| Evaluation | scikit-learn · rouge-score · bert-score |
| Package Manager | **uv** (mandatory) |
| Dataset | Jigsaw Toxic Comments + Synthetic Q&A |
| Deployment | Hugging Face Spaces (Docker/Streamlit) |

---

## 🎥 Demo Video

<video src="assets/demo_video.mp4" controls="controls" muted="muted" width="100%"></video>

> 💡 *If the video player doesn't load automatically, [click here to view the demo video directly](https://github.com/guru22cseb37/DocChat-AI/raw/main/assets/demo_video.mp4).*

---

## 📸 Screenshots

![Document Q&A Empty State](assets/screenshot1.png)
![Document Q&A Results](assets/screenshot2.png)
![Toxicity Checker Mode](assets/screenshot3.png)
![Toxicity Checker Settings](assets/screenshot4.png)

---

## 📝 Design Decisions

**Why TinyLlama over Phi-2?**  
TinyLlama (1.1B) is significantly smaller and runs on CPU without OOM errors. Phi-2 (2.7B) requires more VRAM and is slower on CPU-only machines — making local demos impractical.

**Why LoRA over full fine-tuning?**  
LoRA trains only 0.1% of model parameters, reducing memory usage by ~99% while maintaining output quality. Full fine-tuning of a 1.1B model would require 16GB+ VRAM.

**Why not RAG?**  
The assignment explicitly prohibits RAG and vector databases. Instead, the document text is injected directly into the prompt (`### Document: ... ### Question: ...`), and the model answers from its fine-tuned understanding of the instruction format.

---

## 🤝 Author

Built with ❤️ for the DAIVTECH AI Developer Internship Assignment.

> *"Think like a real AI developer and build a clean, working, locally-runnable system."*  
> — DAIVTECH Assignment Brief

---

<div align="center">

**⭐ Star this repo if it helped you!**

Made with ⚡ by [GURUmsd](https://github.com/guru22cseb37)

</div>
