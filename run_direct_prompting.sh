gpt3_model="gpt-3.5-turbo"
gpt4_model="gpt-4o"
# Path to llama3.1 model, can be a path to huggingface model hub, or a local model dir
llama3_model="/Users/tushar/CancerModels/set-aside/sa-y4/prompt_pdcr_ebi/meta-llama/Meta-Llama-3.1-70B-Instruct"

openai_api_key=""

# Decide which model you are using
#model=$gpt4_model
model=$llama3_model

input_file="fulltext_tokenized_dev.json"
abstract_text_dir="abstract_texts_updated_split/dev"
full_text_dir="articles"
gold_annotation_path="adjudicated_gold/"
output_filename=${model}_70B_full_text_dev_output.json

echo $output_filename

### Run openai model
#python load_openai.py --model_name $model --api_key $openai_api_key --input_file $input_file \
#--output_file $output_filename

### Run llama3.1 model Meta-Llama-3.1-70B-Instruct_70B_dev_output
#/usr/bin/python3 load_llama3.py --model_name $model --input_file $input_file --output_file $output_filename


### Run post processing
python3 parse_llm_output.py --llm_generation $output_filename --parsed_output ${model}_70B_full_text_dev_output_parsed.json \
--information_map $input_file --abstract_text_dir $full_text_dir #$abstract_text_dir


### Run evaluation
#python3 evaluation.py --gold_dir $gold_annotation_path --pred_file ${model}_dev_output_parsed.json
