# imports are here

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

text = "This accelerator is amazingly efficient."

tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print("TOKENS:", tokens)
print("IDS:", ids)

encoded = tokenizer(text)
print(encoded)

print(encoded["input_ids"])
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

print("\n--- padding example (attention_mask can include 0) ---")
batch_texts = [
    "short sentence",
    "this is a much longer sentence than the first one",
]
batch_encoded = tokenizer(batch_texts, padding=True, return_tensors="pt")
print("batch input_ids:\n", batch_encoded["input_ids"])
print("batch attention_mask:\n", batch_encoded["attention_mask"])
print("first tokens:", tokenizer.convert_ids_to_tokens(batch_encoded["input_ids"][0].tolist()))
print("second tokens:", tokenizer.convert_ids_to_tokens(batch_encoded["input_ids"][1].tolist()))