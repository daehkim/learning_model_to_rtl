# imports are here

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "Great performance.",
    "The compiler is functional but some workloads still show disappointign latency behavior."
]

batch_inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=8, # 최대 토큰 수 제한. 그냥 잘려버림. 그래서 메모리/지연/처리량에는 유리하지만 정확도 (의미 보존) 손실이 가능하다.
    # pytorch tensor 형태로 반환
    return_tensors="pt"
)

print("input_ids shape:", batch_inputs["input_ids"].shape)
print("attention_mask shape:", batch_inputs["attention_mask"].shape)
print("batch input_ids:\n", batch_inputs["input_ids"])
print("batch attention_mask:\n", batch_inputs["attention_mask"])
print("first tokens:", tokenizer.convert_ids_to_tokens(batch_inputs["input_ids"][0].tolist()))
print("second tokens:", tokenizer.convert_ids_to_tokens(batch_inputs["input_ids"][1].tolist()))