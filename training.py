import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import torch.nn as nn
from tqdm import tqdm
import os

# set BASE_PATH 
BASE_PATH = "../legal"

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

class CUADDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_labels, max_length=512):
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
            label_index = item['label']
            if 0 <= label_index < self.num_labels:
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
            print(f"\nEpoch {epoch+1}/{epochs}")

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

    def predict(self, model, text, threshold=0.5):
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

def main():
    trainer = LegalBERTTrainer()

    print("Starting LegalBERT Training...")

    model = trainer.train(
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        save_path=f"{BASE_PATH}/trained_legalbert"
    )

    test_text = """
    This Agreement shall be governed by and construed in accordance with the laws
    of the State of Delaware. The parties agree that any disputes arising under this
    Agreement shall be resolved through binding arbitration.
    """

    predictions = trainer.predict(model, test_text)

    print(f"\nTest Prediction:")
    for pred in predictions:
        print(f"  • {pred['clause_type']}: {pred['confidence']:.3f}")

if __name__ == "__main__":
    main()
