import os

# Apple 환경에서 간헐적인 OpenMP/SHM 충돌(bus error) 완화용.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from transformers import AutoTokenizer


def main() -> None:
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    print(f"[1/3] loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "This accelerator is amazingly efficient."
    print("[2/3] tokenizing text", flush=True)
    # 기본 실행은 torch를 쓰지 않는 경로로 유지해서 bus error를 피한다.
    inputs = tokenizer(text)
    print("inputs:", inputs)
    print("num tokens:", len(inputs["input_ids"]))
    print("decoded tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

    print("[3/3] decoding token ids back to text", flush=True)
    print("decoded text:", tokenizer.decode(inputs["input_ids"]))

    if os.environ.get("RUN_INFERENCE") == "1":
        print("[optional] trying pipeline inference", flush=True)
        # NOTE: 환경에 따라 여기서 torch 런타임 bus error가 날 수 있다.
        from transformers import pipeline

        classifier = pipeline("sentiment-analysis", model=model_name, device=-1)
        outputs = classifier(text)
        print("pipeline outputs:", outputs)


if __name__ == "__main__":
    main()