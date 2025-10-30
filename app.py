%%writefile app.py
import streamlit as st
import json, torch, os, PyPDF2, docx, warnings
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
from docx import Document

warnings.filterwarnings('ignore')

# Data Classes
@dataclass
class RiskAssessment:
    clause_type: str
    risk_level: str
    confidence: float
    explanation: str
    recommendation: str

@dataclass
class DocumentAnalysis:
    summary: str
    key_clauses: List[Dict]
    risk_assessments: List[RiskAssessment]
    overall_risk_score: float
    recommendations: List[str]

# LegalBERT 
class LegalBERTMultiLabel(nn.Module):
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", num_labels=41):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class LegalDocumentAnalyzer:
    def __init__(self, model_path="/content/drive/MyDrive/legalnlp/trained_legalbert"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(model_path):
            st.warning(f"Model path {model_path} not found, using untrained model")
            self._init_default_setup()
        else:
            self._load_trained_model()
        self._init_risk_mappings()

    def _init_default_setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.num_labels = 41
        self.model = LegalBERTMultiLabel("nlpaueb/legal-bert-base-uncased", self.num_labels)
        self.model.to(self.device).eval()
        self.idx2label = {str(i): f"Clause_{i}" for i in range(self.num_labels)}

    def _load_trained_model(self):
        try:
            with open(os.path.join(self.model_path, "config.json"), 'r') as f:
                config = json.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.idx2label = config.get("idx2label", {})
            self.num_labels = config.get("num_labels", len(self.idx2label))
            model_name = config.get("model_name", "nlpaueb/legal-bert-base-uncased")
            self.model = LegalBERTMultiLabel(model_name, self.num_labels)
            state_dict = torch.load(os.path.join(self.model_path, "model.pt"), map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device).eval()
            st.success("Trained LegalBERT loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            self._init_default_setup()

    def _init_risk_mappings(self):
        self.risk_mappings = {
            'Uncapped Liability': 'HIGH',
            'Termination For Cause': 'HIGH',
            'Non-Compete': 'HIGH',
            'Exclusivity': 'MEDIUM',
            'Cap On Liability': 'LOW',
            'Governing Law': 'LOW'
        }

    def extract_text(self, file_path: str) -> str:
        ext = file_path.lower().split('.')[-1]
        if ext == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join([p.extract_text() for p in reader.pages])
        elif ext == "docx":
            doc_file = docx.Document(file_path)
            return "\n".join([p.text for p in doc_file.paragraphs])
        elif ext == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def chunk_text(self, text: str, chunk_size=400, overlap=100):
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= len(words): break
            start = end - overlap
        return chunks

    def predict(self, text, threshold=0.5):
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device))
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
        return [{"clause_type": self.idx2label.get(str(i), f"Clause_{i}"), "confidence": float(p)}
                for i, p in enumerate(probs) if p > threshold]

    def analyze_document(self, file_path: str) -> DocumentAnalysis:
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text)
        all_preds = []
        for c in chunks:
            all_preds.extend(self.predict(c, threshold=0.3))
        unique = {p["clause_type"]: p for p in all_preds}
        clauses = list(unique.values())
        risk_assessments = [RiskAssessment(
            clause_type=c["clause_type"],
            risk_level=self.risk_mappings.get(c["clause_type"], "MEDIUM"),
            confidence=c["confidence"],
            explanation=f"{c['clause_type']} needs review.",
            recommendation=f"Check {c['clause_type']} carefully."
        ) for c in clauses]
        return DocumentAnalysis(
            summary=f"Found {len(clauses)} clauses.",
            key_clauses=clauses,
            risk_assessments=risk_assessments,
            overall_risk_score=len([r for r in risk_assessments if r.risk_level=='HIGH']),
            recommendations=["Consult legal counsel"]
        )

# Gemma Synthesizer
class GemmaLegalSynthesizer:
    def __init__(self, model_name="google/gemma-2-2b-it", use_4bit=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ) if use_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_response(self, prompt, max_new_tokens=400):
        try:
            messages = [{"role": "user", "content": prompt}]
            input_data = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **input_data,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generated_ids = outputs[0][input_data['input_ids'].shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Error generating analysis. Please try again."

# Streamlit UI 
st.title("AI-Powered Contract Analysis System")

uploaded = st.file_uploader("Upload a contract file (PDF/DOCX/TXT):", type=["pdf","docx","txt"])

if uploaded:
    tmp_path = os.path.join("/content", uploaded.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    st.info("Running LegalBERT clause detection...")
    analyzer = LegalDocumentAnalyzer()
    lb_result = analyzer.analyze_document(tmp_path)
    st.success(lb_result.summary)

    st.write("### 📋 Detected Clauses")
    for c in lb_result.key_clauses:
        st.write(f"- **{c['clause_type']}** (Confidence: {c['confidence']:.2f})")

    st.info("Generating Gemma 2 summary & risk analysis...")
    gemma = GemmaLegalSynthesizer()
    text = analyzer.extract_text(tmp_path)
    report = gemma.generate_response(f"Summarize this legal document:\n\n{text[:2000]}")

    st.subheader("📄 Executive Summary")
    st.write(report)

    st.download_button("Download Results as JSON", json.dumps(lb_result.__dict__, default=lambda o:o.__dict__), file_name="contract_analysis.json")

    # Chatbot query section
    st.markdown("---")
    st.subheader("Ask Questions About This Contract")
    user_query = st.text_input("Type your question (e.g., 'What are the high-risk clauses?')")

    if user_query:
        with st.spinner("Gemma is analyzing your question..."):
            context = analyzer.extract_text(tmp_path)
            chat_prompt = f"You are a legal assistant. Based on the following contract:\n\n{context[:3000]}\n\nAnswer this question clearly and accurately:\n{user_query}"
            response = gemma.generate_response(chat_prompt, max_new_tokens=400)
        st.write(response)
