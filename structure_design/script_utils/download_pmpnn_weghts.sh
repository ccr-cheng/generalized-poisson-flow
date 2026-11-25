#!/bin/bash
set -e

echo "=== Checking ProteinMPNN weights ==="

cd ProteinMPNN

if [ -d "ca_model_weights" ] && [ "$(ls -A ca_model_weights)" ]; then
    echo "[skip] ca_model_weights exists."
else
    echo "[download] downloading ca_model_weights..."
    rm -rf ca_model_weights
    mkdir ca_model_weights
    cd ca_model_weights
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_002.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_010.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_020.pt
    cd ..
fi

if [ -d "vanilla_model_weights" ] && [ "$(ls -A vanilla_model_weights)" ]; then
    echo "[skip] vanilla_model_weights exists."
else
    echo "[download] downloading vanilla_model_weights..."
    rm -rf vanilla_model_weights
    mkdir vanilla_model_weights
    cd vanilla_model_weights
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_002.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_010.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_020.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_030.pt
    cd ..
fi

cd ..

echo "=== Downloading ESMFold into HF cache ($HF_HOME) ==="

python <<'EOF'
from huggingface_hub import snapshot_download

print("[info] Using HF cache at:", __import__("os").environ.get("HF_HOME", "~/.cache/huggingface"))
snapshot_download(
    repo_id="facebook/esmfold_v1",
    local_dir_use_symlinks=False,
)
print("[done] ESMFold is now cached.")
EOF

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Done, set HF_OFFLINE = 1 ==="
