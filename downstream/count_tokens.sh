OUTPUT_ROOT="dataset_cnt_stat"
TOKENIZER_NAME="KETI-AIR/long-ke-t5-small"
HF_CACHE_DIR="../huggingface_datasets"
HF_DATA_DIR="../data/downstreams"
TOKENIZER_NAME_OUT="KETI-AIR_long-ke-t5-small"

# # summarization
# DATASET_NAME=cnn_dailymail
# DATASET_CONFIG_NAME="3.0.0"
# INPUT_COLUMN="article"
# TARGET_COLUMN="highlights"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME=big_patent
# DATASET_CONFIG_NAME="all"
# INPUT_COLUMN="description"
# TARGET_COLUMN="abstract"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME=multi_news
# INPUT_COLUMN="document"
# TARGET_COLUMN="summary"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

# DATASET_NAME="ccdv/pubmed-summarization"
# DATASET_CONFIG_NAME="section"
# INPUT_COLUMN="article"
# TARGET_COLUMN="abstract"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="ccdv/arxiv-summarization"
# DATASET_CONFIG_NAME="section"
# INPUT_COLUMN="article"
# TARGET_COLUMN="abstract"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="ccdv/mediasum"
# DATASET_CONFIG_NAME="roberta"
# INPUT_COLUMN="document"
# TARGET_COLUMN="summary"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="ccdv/WCEP-10"
# DATASET_CONFIG_NAME="roberta"
# INPUT_COLUMN="document"
# TARGET_COLUMN="summary"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_summary_and_report"
# INPUT_COLUMN="passage"
# TARGET_COLUMN="generative_summary"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

# DATASET_NAME="KETI-AIR/nikl_summarization"
# DATASET_CONFIG_NAME="base"
# INPUT_COLUMN="article"
# TARGET_COLUMN="summary_sentences"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_book_summarization"
# DATASET_CONFIG_NAME="base"
# INPUT_COLUMN="passage"
# TARGET_COLUMN="summary"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_dialog_summarization"
# DATASET_CONFIG_NAME="roberta_prepended"
# INPUT_COLUMN="dialog"
# TARGET_COLUMN="summary"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_document_summarization"
# DATASET_CONFIG_NAME="default"
# INPUT_COLUMN="passage"
# TARGET_COLUMN="abstractive"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_document_summarization"
# DATASET_CONFIG_NAME="law"
# INPUT_COLUMN="passage"
# TARGET_COLUMN="abstractive"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_document_summarization"
# DATASET_CONFIG_NAME="magazine"
# INPUT_COLUMN="passage"
# TARGET_COLUMN="abstractive"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_document_summarization"
# DATASET_CONFIG_NAME="news"
# INPUT_COLUMN="passage"
# TARGET_COLUMN="abstractive"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_paper_summarization"
# DATASET_CONFIG_NAME="default"
# INPUT_COLUMN="original_text"
# TARGET_COLUMN="summary_text"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_paper_summarization"
# DATASET_CONFIG_NAME="paper_entire"
# INPUT_COLUMN="original_text"
# TARGET_COLUMN="summary_text"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_paper_summarization"
# DATASET_CONFIG_NAME="paper_section"
# INPUT_COLUMN="original_text"
# TARGET_COLUMN="summary_text"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_paper_summarization"
# DATASET_CONFIG_NAME="patent_entire"
# INPUT_COLUMN="original_text"
# TARGET_COLUMN="summary_text"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_paper_summarization"
# DATASET_CONFIG_NAME="patent_section"
# INPUT_COLUMN="original_text"
# TARGET_COLUMN="summary_text"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# # MRC

# DATASET_NAME="squad"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

# DATASET_NAME="squad_v2"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

# DATASET_NAME="squad_kor_v1"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

# DATASET_NAME="squad_kor_v2"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

# DATASET_NAME="KETI-AIR/aihub_news_mrc"
# DATASET_CONFIG_NAME="squad.v1.like"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_news_mrc"
# DATASET_CONFIG_NAME="squad.v2.like"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_admin_docs_mrc"
# DATASET_CONFIG_NAME="squad.v1.like"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}

# DATASET_NAME="KETI-AIR/aihub_admin_docs_mrc"
# DATASET_CONFIG_NAME="squad.v2.like"
# INPUT_COLUMN="context"
# TARGET_COLUMN="question"
# python count_tokens.py \
# --dataset_name ${DATASET_NAME} \
# --dataset_config_name ${DATASET_CONFIG_NAME} \
# --hf_cache_dir ${HF_CACHE_DIR} \
# --hf_data_dir ${HF_DATA_DIR} \
# --tokenizer_name ${TOKENIZER_NAME} \
# --input_column ${INPUT_COLUMN} \
# --target_column ${TARGET_COLUMN} \
# --output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}/${DATASET_CONFIG_NAME}


# Translation
DATASET_NAME="KETI-AIR/aihub_koenzh_food_translation"
INPUT_COLUMN="ko"
TARGET_COLUMN="en"
python count_tokens.py \
--dataset_name ${DATASET_NAME} \
--hf_cache_dir ${HF_CACHE_DIR} \
--hf_data_dir ${HF_DATA_DIR} \
--tokenizer_name ${TOKENIZER_NAME} \
--input_column ${INPUT_COLUMN} \
--target_column ${TARGET_COLUMN} \
--is_translation \
--output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

DATASET_NAME="KETI-AIR/aihub_scitech_translation"
INPUT_COLUMN="ko"
TARGET_COLUMN="en"
python count_tokens.py \
--dataset_name ${DATASET_NAME} \
--hf_cache_dir ${HF_CACHE_DIR} \
--hf_data_dir ${HF_DATA_DIR} \
--tokenizer_name ${TOKENIZER_NAME} \
--input_column ${INPUT_COLUMN} \
--target_column ${TARGET_COLUMN} \
--is_translation \
--output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

DATASET_NAME="KETI-AIR/aihub_scitech20_translation"
INPUT_COLUMN="ko"
TARGET_COLUMN="en"
python count_tokens.py \
--dataset_name ${DATASET_NAME} \
--hf_cache_dir ${HF_CACHE_DIR} \
--hf_data_dir ${HF_DATA_DIR} \
--tokenizer_name ${TOKENIZER_NAME} \
--input_column ${INPUT_COLUMN} \
--target_column ${TARGET_COLUMN} \
--is_translation \
--output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

DATASET_NAME="KETI-AIR/aihub_socialtech20_translation"
INPUT_COLUMN="ko"
TARGET_COLUMN="en"
python count_tokens.py \
--dataset_name ${DATASET_NAME} \
--hf_cache_dir ${HF_CACHE_DIR} \
--hf_data_dir ${HF_DATA_DIR} \
--tokenizer_name ${TOKENIZER_NAME} \
--input_column ${INPUT_COLUMN} \
--target_column ${TARGET_COLUMN} \
--is_translation \
--output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

DATASET_NAME="KETI-AIR/aihub_spoken_language_translation"
INPUT_COLUMN="ko"
TARGET_COLUMN="en"
python count_tokens.py \
--dataset_name ${DATASET_NAME} \
--hf_cache_dir ${HF_CACHE_DIR} \
--hf_data_dir ${HF_DATA_DIR} \
--tokenizer_name ${TOKENIZER_NAME} \
--input_column ${INPUT_COLUMN} \
--target_column ${TARGET_COLUMN} \
--is_translation \
--output_dir ${OUTPUT_ROOT}/${TOKENIZER_NAME_OUT}/${DATASET_NAME}

