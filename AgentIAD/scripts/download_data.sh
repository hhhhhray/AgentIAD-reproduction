#!/bin/bash
# Download MMAD dataset from Hugging Face
# Run this script from the AgentIAD project root directory

set -e

echo "=== Downloading MMAD Dataset ==="

DATA_DIR="./data/MMAD"
mkdir -p "$DATA_DIR"

# Option 1: Download ZIP from Hugging Face
echo "Downloading ALL_DATA.zip from Hugging Face..."
wget -O "${DATA_DIR}/ALL_DATA.zip" \
    "https://huggingface.co/datasets/jiang-cc/MMAD/resolve/refs%2Fpr%2F1/ALL_DATA.zip?download=true"

echo "Extracting dataset..."
cd "$DATA_DIR"
unzip ALL_DATA.zip
rm ALL_DATA.zip
cd -

# Download domain_knowledge.json from the MMAD GitHub repo
echo "Downloading domain_knowledge.json..."
wget -O "${DATA_DIR}/domain_knowledge.json" \
    "https://raw.githubusercontent.com/jam-cc/MMAD/main/dataset/MMAD/domain_knowledge.json"

echo "=== Dataset download complete ==="
echo "Dataset directory: ${DATA_DIR}"
ls -la "$DATA_DIR"
