#!/usr/bin/env bash
# Step 2 — llama.cpp: build once, then run a small GGUF locally.
# Official repo: https://github.com/ggerganov/llama.cpp
set -euo pipefail

echo "1) Clone llama.cpp next to this repo (or anywhere you prefer):"
echo "   git clone https://github.com/ggerganov/llama.cpp"
echo ""
echo "2) Build (pick one backend; CPU is simplest):"
echo "   cd llama.cpp && cmake -B build && cmake --build build -j"
echo ""
echo "3) Download a small GGUF (example family; pick a specific file from TheBloke or upstream):"
echo "   # Place under learning_model_to_rtl/models/ (gitignored) or outside the repo"
echo ""
echo "4) Run interactive inference:"
echo "   ./build/bin/llama-cli -m /path/to/model.gguf -p \"Why does KV cache matter?\" -n 64"
echo ""
echo "5) Optional: server / OpenAI-compatible API — see llama.cpp README server section."
