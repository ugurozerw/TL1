from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch

# Cihazı ayarlayın
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT Tokenizer ve modelini yükle
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)

# Örnek veriler
texts = ["This is a great movie!", "This is a terrible movie!"]
labels = [1, 0]  # 1: pozitif, 0: negatif

# Tokenization ve DataLoader hazırlığı
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# TensorDataset ve DataLoader oluştur
dataset = TensorDataset(input_ids, attention_mask, torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=2)

# Modeli değerlendirme moduna al
model.eval()

# Verileri modelden geçir ve tahminleri al
with torch.no_grad():
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print("Predictions:", predictions)

# Modeli fine-tuning için hazırla (eğitim moduna al)
model.train()

# Fine-tuning için eğitim döngüsü burada eklenmelidir (öğretici amaçlı atlanmıştır)
