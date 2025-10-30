# @title
import os
import json
import torch
import numpy as np
import torch
from sklearn.metrics import classification_report, hamming_loss, jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import PyPDF2  # for PDF support
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torch.nn as nn


# Ppaths
BASE_PATH = "../legalnlp"
DATA_PATH = os.path.join(BASE_PATH, "processed_cuad")
MODEL_PATH = os.path.join(BASE_PATH, "trained_legalbert")

# Define LegalBERTMultiLabel and CUADDataset classes again
class LegalBERTMultiLabel(nn.Module):
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased", num_labels=41):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_depth, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class CUADDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_labels, max_length=512): # Added num_labels
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.num_labels = num_labels 
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        # FIX: Use 'label' key and convert single int label to one-hot encoding
        label_index = item['label']
        labels = torch.zeros(self.num_labels) 
        labels[label_index] = 1.0


        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }


# Define LegalBERTTrainer class again
class LegalBERTTrainer:
    def __init__(self, data_dir=f"{BASE_PATH}/processed_cuad",
                 model_name="nlpaueb/legal-bert-base-uncased"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        with open(f"{data_dir}/label_mappings.json", 'r') as f:
            mappings = json.load(f)

        self.label2idx = mappings['label2idx']
        self.idx2label = mappings['idx2label']
        self.num_labels = len(self.label2idx)

        print(f"Initialized trainer")
        print(f"Device: {self.device}")
        print(f"Labels: {self.num_labels}")

    def create_dataloaders(self, batch_size=16):
        # Pass num_labels to CUADDataset
        train_dataset = CUADDataset(f"{self.data_dir}/train.json", self.tokenizer, self.num_labels)
        val_dataset = CUADDataset(f"{self.data_dir}/val.json", self.tokenizer, self.num_labels)
        test_dataset = CUADDataset(f"{self.data_dir}/test.json", self.tokenizer, self.num_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def calculate_class_weights(self):
        with open(f"{self.data_dir}/train.json", 'r') as f:
            train_data = json.load(f)

        label_counts = np.zeros(self.num_labels)
        for item in train_data:
            # FIX: Use 'label' key
            label_index = item['label']
            if 0 <= label_index < self.num_labels:
                # Convert the single label index to a one-hot encoded vector
                one_hot_label = np.zeros(self.num_labels)
                one_hot_label[label_index] = 1
                label_counts += one_hot_label
            else:
                print(f"Warning: Invalid label index {label_index} found in training data.")


        total_samples = len(train_data)
        pos_weights = []

        for count in label_counts:
            if count > 0:
                neg_count = total_samples - count
                pos_weight = neg_count / count
                pos_weights.append(pos_weight)
            else:
                pos_weights.append(1.0)

        return torch.FloatTensor(pos_weights).to(self.device)

    def train(self, epochs=3, batch_size=16, learning_rate=2e-5,
              save_path=f"{BASE_PATH}/trained_legalbert"):

        train_loader, val_loader, _ = self.create_dataloaders(batch_size)

        model = LegalBERTMultiLabel(self.model_name, self.num_labels)
        model.to(self.device)

        pos_weights = self.calculate_class_weights()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        print(f"\nTraining Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\n Epoch {epoch+1}/{epochs}")

            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc="Training")

            for batch in train_pbar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = train_loss / len(train_loader)

            model.eval()
            val_loss = 0

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation")
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_val_loss = val_loss / len(val_loader)

            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(model, save_path)
                print(f"   Saved best model (val_loss: {avg_val_loss:.4f})")

        print(f"\nTraining complete!")
        return model

    def save_model(self, model, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/model.pt")
        self.tokenizer.save_pretrained(save_path)

        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label
        }
        with open(f"{save_path}/config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def load_model(self, model_path):
        with open(f"{model_path}/config.json", 'r') as f:
            config = json.load(f)

        model = LegalBERTMultiLabel(config['model_name'], config['num_labels'])
        model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=self.device))
        model.to(self.device)
        return model

    def predict(self, text, threshold=0.5):
        model = self.load_model(MODEL_PATH) # Load the model before prediction
        model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        predictions = []
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                clause_type = self.idx2label[str(i)]
                predictions.append({
                    'clause_type': clause_type,
                    'confidence': float(prob)
                })

        return sorted(predictions, key=lambda x: x['confidence'], reverse=True)


class ModelEvaluator:
    def __init__(self, trainer, model_path=MODEL_PATH, batch_size=16):
        self.trainer = trainer
        self.device = trainer.device

        # Load trained model
        self.model = trainer.load_model(model_path)
        _, _, self.test_loader = trainer.create_dataloaders(batch_size=batch_size)

    def evaluate_model(self, threshold=0.5):
        print(" Evaluating Model Performance...")
        self.model.eval()
        all_predictions, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > threshold).astype(int)

                all_predictions.extend(preds)
                all_labels.extend(labels.numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Overall metrics
        hamming = hamming_loss(all_labels, all_predictions)
        jaccard = jaccard_score(all_labels, all_predictions, average='samples')
        print(f"Hamming Loss: {hamming:.4f}, Jaccard Score: {jaccard:.4f}")

        # Per-class
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=self.trainer.idx2label.values(),
            output_dict=True, zero_division=0
        )

        # Top/worst performing
        class_f1_scores = [(name, metrics['f1-score'])
                           for name, metrics in class_report.items()
                           if isinstance(metrics, dict) and 'f1-score' in metrics]
        class_f1_scores.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 10 Best Performing Clause Types:")
        for i, (clause, f1) in enumerate(class_f1_scores[:10]):
            print(f"  {i+1}. {clause}: F1={f1:.3f}")
        print("\nWorst 5 Performing Clause Types:")
        for i, (clause, f1) in enumerate(class_f1_scores[-5:]):
            print(f"  {clause}: F1={f1:.3f}")

        return all_predictions, all_labels, class_report

    def find_optimal_threshold(self):
        print("\nFinding Optimal Thresholds...")
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Getting probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask)
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        thresholds = np.arange(0.1, 0.9, 0.1)
        optimal_thresholds = []

        for c in range(all_labels.shape[1]):
            best_f1, best_thresh = 0, 0.5
            for t in thresholds:
                preds = (all_probs[:, c] > t).astype(int)
                tp = np.sum((preds == 1) & (all_labels[:, c] == 1))
                fp = np.sum((preds == 1) & (all_labels[:, c] == 0))
                fn = np.sum((preds == 0) & (all_labels[:, c] == 1))
                if tp + fp > 0 and tp + fn > 0:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = t
            optimal_thresholds.append(best_thresh)

        return optimal_thresholds

    def test_on_sample_contracts(self):
        samples = {
            "Governing Law": "This Agreement shall be governed by and construed in accordance with the laws of California.",
            "Termination": "Either party may terminate this Agreement immediately upon written notice if the other party materially breaches any provision of this Agreement.",
            "IP Assignment": "Employee hereby assigns to Company all right, title, and interest in inventions made during employment.",
            "Liability Cap": "Neither party's liability under this Agreement shall exceed the total amount paid by Customer in the preceding twelve months."
        }

        print("\nTesting on Sample Contracts:")
        for expected, text in samples.items():
            preds = self.trainer.predict(self.model, text)
            print(f"\nExpected: {expected}")
            print(f"Text: {text[:100]}...")
            print("Predictions:")
            if preds:
                for i, p in enumerate(preds[:3]):
                    emoji = "🟢" if p['confidence']>0.7 else "🟡" if p['confidence']>0.5 else "🔴"
                    print(f"  {i+1}. {emoji} {p['clause_type']}: {p['confidence']:.3f}")
            else:
                print(" No predictions above threshold")

    def test_on_file(self, file_path, top_k=5, chunk_size=500):
        """
        Test model on a contract file (.txt or .pdf) with chunking.
        Shows top_k predictions from the entire document, merging duplicates.
        """
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            return

        # Extract text
        contract_text = ""
        if file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                contract_text = f.read().strip()
        elif file_path.endswith(".pdf"):
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        contract_text += page_text + "\n"
            contract_text = contract_text.strip()
        else:
            print(" Unsupported file type. Use .txt or .pdf")
            return

        print(f"\nTesting on File: {file_path}")
        print("="*60)
        print(f"Contract text (first 300 chars): {contract_text[:300]}...\n")

        # Chunk the text
        words = contract_text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        # Aggregate predictions from all chunks
        clause_confidences = {}
        for chunk in chunks:
            preds = self.trainer.predict(self.model, chunk)
            for pred in preds:
                clause_type = pred['clause_type']
                conf = pred['confidence']
                # Keep the max confidence for each clause type
                if clause_type not in clause_confidences or conf > clause_confidences[clause_type]:
                    clause_confidences[clause_type] = conf

        # Sort predictions by confidence
        sorted_predictions = sorted(clause_confidences.items(), key=lambda x: x[1], reverse=True)

        print(f" Top {top_k} Predictions for entire document:")
        if sorted_predictions:
            for i, (clause_type, conf) in enumerate(sorted_predictions[:top_k]):
                confidence_emoji = "🟢" if conf > 0.7 else "🟡" if conf > 0.5 else "🔴"
                print(f"   {i+1}. {confidence_emoji} {clause_type}: {conf:.3f}")
        else:
            print(" No predictions above threshold")


if __name__ == "__main__":
    # Initialize trainer
    trainer = LegalBERTTrainer(data_dir=DATA_PATH)

    # Initialize evaluator (loads model automatically)
    evaluator = ModelEvaluator(trainer)

    # Evaluate
    evaluator.evaluate_model()
    evaluator.find_optimal_threshold()
    evaluator.test_on_sample_contracts()
