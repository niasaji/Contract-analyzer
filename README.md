# AI-Powered Contract Analysis System

An **AI-driven legal document analysis tool** that automates clause detection, risk evaluation, summarization, and Q&A over uploaded contracts.  
Built with **LegalBERT** for multi-label clause classification and **Gemma 2** for legal text summarization and question answering — all wrapped in an intuitive **Streamlit interface**.

---

## Features

- **File Upload Support** — Accepts PDF, DOCX, and TXT contract files.
- **Clause Detection (LegalBERT)** — Identifies key clauses such as confidentiality, non-compete, and governing law.
- **Risk Assessment** — Classifies clauses as HIGH, MEDIUM, or LOW risk using pre-mapped legal logic.
- **Legal Summarization (Gemma 2)** — Generates concise executive summaries of uploaded contracts.
- **Interactive Q&A** — Users can ask natural language questions about the contract (e.g., “Who are the parties involved?”).
- **Export Results** — Download full analysis in JSON format.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| UI | Streamlit |
| NLP Models | LegalBERT (`nlpaueb/legal-bert-base-uncased`), Gemma 2 (`google/gemma-2-2b-it`) |
| Backend | PyTorch, Transformers |
| File Handling | PyPDF2, python-docx |
| Quantization | BitsAndBytes (for 4-bit model loading) |

---

## Project Architecture

app.py
│
├── LegalBERTMultiLabel          # Multi-label classification model
├── LegalDocumentAnalyzer         # Handles text extraction, clause detection, risk scoring
├── GemmaLegalSynthesizer         # Generates summaries and Q&A responses
└── Streamlit UI                  # User interface for upload, display, and chat

#### Setup & Installation
1️. Clone the Repository
```bash
git clone https://github.com/<your-username>/ai-contract-analyzer.git
cd ai-contract-analyzer
```
2️. Install Dependencies and Authentication Tokens
Make sure you’re in a Python 3.10+ environment, then install the requirements:
```bash
!pip install transformers torch scikit-learn tqdm PyPDF2 python-docx -U accelerate bitsandbytes torchvision streamlit pyngrok
```
Ngrok token auth
```bash
from pyngrok import ngrok
ngrok.set_auth_token("your_auth_token")
```
HuggingFace token auth
```bash
from huggingface_hub import login
login("your_auth_token")
```

3️. (Optional) Connect Google Drive
If running on Google Colab, mount your drive to load trained models:

```python
from google.colab import drive
drive.mount('/content/drive')
```
4️. Run the App
```bash
!streamlit run app.py --server.port 8501
```
If using Colab, you can expose the app using:

```bash
!streamlit run app.py &>/content/logs.txt &
public_url = ngrok.connect(8501)
print("Streamlit app running at:", public_url)
```

#### How It Works

1. Upload a Contract - Accepts .pdf, .docx, or .txt files.
2. Clause Detection - LegalBERT performs clause tagging with confidence scores.
3. Risk Mapping - Each clause is assigned a risk level based on pre-defined mappings.
4. Summarization - Gemma 2 produces an executive summary.
5. Interactive Q&A - Users can query the analyzed text directly via natural language.



## Research & Applications
This project addresses research gaps in:
- Automated Legal Clause Detection using transformer-based contextual embeddings.
- Explainable AI in Law via interpretable risk mappings.
- Natural Language Interfaces for Legal Analysis through LLM-based Q&A.
- Lightweight Model Deployment using 4-bit quantization for on-device inference.
