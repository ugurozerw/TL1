from transformers import pipeline

# 'sentiment-analysis' pipeline'ını yükle
classifier = pipeline('sentiment-analysis')

# Test edilecek metinler
texts = ["I love this!", "I hate this!", "This is fantastic!", "This is terrible!"]

# Metinleri sınıflandır ve sonuçları yazdır
results = classifier(texts)
for text, result in zip(texts, results):
    print(f"'{text}' sentiment: {result['label']} with a score of {result['score']:.4f}")
