import os
import json
from transformers import AutoTokenizer

BASE_PATH = "../legal"
RAW_DATA_PATH = f"{BASE_PATH}/CUAD_v1.json"
OUTPUT_DIR = f"{BASE_PATH}/processed_cuad"

# Hugging Face cache (optional, saves redownloads)
os.environ["TRANSFORMERS_CACHE"] = f"{BASE_PATH}/hf_cache"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Clause labels (41 CUAD types)
clause_labels = [
    "Affiliates", "Anti-Assignment", "Anti-Embarrassment", "Arbitration",
    "Audits", "Cap On Liability", "Change Of Control", "Confidentiality",
    "Consequential Damages Waiver", "Covenant Not To Sue", "Damages",
    "Data Security", "Disclosure Of Contract", "Dispute Resolution",
    "Exclusivity", "Expiration Date", "Force Majeure", "Governing Law",
    "Indemnification", "Insurance", "IP Ownership", "Joint Venture",
    "Jurisdiction", "Limitation Of Liability", "Liquidated Damages",
    "Most Favored Nation", "Non-Compete", "Non-Disparagement",
    "Non-Solicit", "Notice Period To Terminate Renewal", "Payments",
    "Price Restrictions", "Renewal Term", "Revenue/Profit Sharing",
    "Right Of First Refusal", "Source Code Escrow", "Subcontracting",
    "Termination For Convenience", "Termination For Insolvency",
    "Third Party Beneficiaries", "Warranty"
]

label2idx = {label: idx for idx, label in enumerate(clause_labels)}
idx2label = {idx: label for label, idx in label2idx.items()}

# Question → clause mapping (simple keyword matching)
def map_question_to_clause_type(question):
    q = question.lower()
    if "terminate" in q: return "Termination For Convenience"
    if "governing law" in q: return "Governing Law"
    if "jurisdiction" in q: return "Jurisdiction"
    if "confidential" in q: return "Confidentiality"
    if "non-compete" in q: return "Non-Compete"
    if "indemnification" in q: return "Indemnification"
    if "arbitration" in q: return "Arbitration"
    if "force majeure" in q: return "Force Majeure"
    if "insurance" in q: return "Insurance"
    if "audit" in q: return "Audits"
    if "expiration" in q or "expire" in q: return "Expiration Date"
    return None

# Load CUAD
with open(RAW_DATA_PATH, "r") as f:
    raw_data = json.load(f)

examples = []
for doc in raw_data["data"]:
    context = doc["paragraphs"][0]["context"]
    if not context or len(context) < 50:
        continue

    for qa in doc["paragraphs"][0]["qas"]:
        clause_type = map_question_to_clause_type(qa["question"])
        if clause_type:
            examples.append({
                "text": context,
                "label": label2idx[clause_type]
            })

print(f"Collected {len(examples)} examples.")

# Train/Val/Test split 
train_split = int(0.8 * len(examples))
val_split = int(0.9 * len(examples))
train_data = examples[:train_split]
val_data = examples[train_split:val_split]
test_data = examples[val_split:]

# Save processed files
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(f"{OUTPUT_DIR}/val.json", "w") as f:
    json.dump(val_data, f, indent=2)

with open(f"{OUTPUT_DIR}/test.json", "w") as f:
    json.dump(test_data, f, indent=2)

with open(f"{OUTPUT_DIR}/label_mappings.json", "w") as f:
    json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

print(f"Saved train/val/test splits and label mappings in {OUTPUT_DIR}")
