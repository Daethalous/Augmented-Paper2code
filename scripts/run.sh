# export OPENAI_API_KEY=""

GPT_VERSION="o3-mini"

PAPER_NAME="Token-level-Direct-Preference-Optimization"
PDF_PATH="../examples/Token-level-Direct-Preference-Optimization.pdf" # .pdf
PDF_JSON_PATH="../examples/Token-level-Direct-Preference-Optimization.json" # .json
PDF_JSON_CLEANED_PATH="../examples/Token-level-Direct-Preference-Optimization_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/Token-level-Direct-Preference-Optimization_origin"
OUTPUT_REPO_DIR="../outputs/Token-level-Direct-Preference-Optimization_repo_origin"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

python ../codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}


python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \
