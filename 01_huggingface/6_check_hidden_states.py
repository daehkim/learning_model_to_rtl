import os
import faulthandler

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Run:\n"
        "  source ../activate_venv.sh\n"
        "  pip install -r requirements.txt\n"
        "  python3 6_check_hidden_states.py"
    ) from exc


def main() -> None:
    faulthandler.enable()
    model_name = os.environ.get(
        "MODEL_NAME",
        "hf-internal-testing/tiny-random-bert",
    )
    text = "This accelerator is amazingly efficient."

    print(f"[1/3] loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[2/3] loading model: {model_name}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False

    inputs = tokenizer(text, return_tensors="pt")
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    print("[3/3] forward with output_hidden_states=True", flush=True)
    with torch.inference_mode():
        outputs = model(**model_inputs, output_hidden_states=True)

    hs = outputs.hidden_states
    print("hidden_states type:", type(hs))
    print("num tensors (embedding + layers):", len(hs))
    for i, h in enumerate(hs):
        print(f"  hidden_states[{i}] shape: {tuple(h.shape)}")
    print("last layer shape:", tuple(hs[-1].shape))


if __name__ == "__main__":
    main()
