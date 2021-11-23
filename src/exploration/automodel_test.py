from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = ["This is the second best day of my life.", "Are you freaking kidding me right now?"]

tokens = tokenizer(raw_inputs, padding=True, return_tensors="pt")
print(tokens)

raw_outputs = model(**tokens)
print(raw_outputs.logits)

predictions = torch.nn.functional.softmax(raw_outputs.logits, dim=-1)
print(predictions)

# max value, index of max value, and corresponding label
labels = model.config.id2label
max_value_index = [(torch.max(p), torch.argmax(p)) for p in predictions] 
[print("{:.5f}".format(e[0].item()),labels[e[1].item()]) for e in max_value_index]
