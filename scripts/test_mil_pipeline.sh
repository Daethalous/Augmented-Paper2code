#!/bin/bash

# Navigate to the workspace root
cd "$(dirname "$0")/.."

# Configuration
PAPER_NAME="MILTimeSeriesClassification"
GPT_VERSION="gpt-5-mini" 
INPUT_JSON="examples/MILTimeSeriesClassification.json"
REPO_JSON="examples/test_repo_urls.json"
OUTPUT_DIR="outputs/tests/MILTimeSeries"

# Ensure the OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[Error] Please set your OPENAI_API_KEY environment variable."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_REPO_DIR="${OUTPUT_DIR}_repo"
mkdir -p "$OUTPUT_REPO_DIR"

# echo "======================================================"
# echo "Step 1: Running 0.1.1 Semantic Parsing"
# echo "======================================================"
# python codes/0.1.1_semantic_parsing.py \
#     --input_json_path "$INPUT_JSON" \
#     --output_path "$OUTPUT_DIR/semantic_parsing_output.json" \
#     --gpt_version "$GPT_VERSION"

# echo ""
# echo "======================================================"
# echo "Step 2: Running 1_planning"
# echo "======================================================"
# python codes/1_planning.py \
#     --paper_name "$PAPER_NAME" \
#     --gpt_version "$GPT_VERSION" \
#     --paper_format "JSON" \
#     --pdf_json_path "$INPUT_JSON" \
#     --repo_json_path "$REPO_JSON" \
#     --output_dir "$OUTPUT_DIR"

# echo ""
# echo "======================================================"
# echo "Step 3: Running 1.1_extract_config"
# echo "======================================================"
# python codes/1.1_extract_config.py \
#     --paper_name "$PAPER_NAME" \
#     --output_dir "$OUTPUT_DIR"

# cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"    

# echo ""
# echo "======================================================"
# echo "Step 4: Running 2_analyzing"
# echo "======================================================"
# python codes/2_analyzing.py \
#     --paper_name "$PAPER_NAME" \
#     --gpt_version "$GPT_VERSION" \
#     --paper_format "JSON" \
#     --pdf_json_path "$INPUT_JSON" \
#     --output_dir "$OUTPUT_DIR"

echo ""
echo "======================================================"
echo "Step 5: Running 3_coding"
echo "======================================================"
python codes/3_coding.py \
    --paper_name "$PAPER_NAME" \
    --gpt_version "$GPT_VERSION" \
    --paper_format "JSON" \
    --pdf_json_path "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --output_repo_dir "${OUTPUT_REPO_DIR}"


echo ""
echo "======================================================"
echo "Testing completed. Results are saved in: $OUTPUT_DIR"
echo "======================================================"
