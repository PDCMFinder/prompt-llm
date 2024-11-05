import torch
import json
import argparse

from Bio.Phylo.NewickIO import tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PreTrainedTokenizer, BitsAndBytesConfig

BitsAndBytesConfig

model_id = "/Users/tushar/CancerModels/set-aside/sa-y4/prompt_pdcr_ebi/meta-llama/" + "Meta-Llama-3.1-8B-Instruct"
PROMPT = "You will be given a sentence from a paper on PDCM. Please extract entities as defined below and return as an XML." \
         " In the XML, please mark the start and end of the entity with <entity_type></entity_type>. " \
         "Please return one entity at one time. That is, if there are n entities in the sentence, " \
         "print the sentence with the marked entity for n times. Do not change anything else in the sentence."
user_input_start = "<|start_header_id|>user<|end_header_id|>\n\n"
assistant_input_start = "<|start_header_id|>assistant<|end_header_id|>\n\n"
user_input_end = "<|eot_id|>\n"


def prepare_examples(input_file):
    """
    Please make sure to modify the code when using other LLAMA models,
    they might have different prompt templates
    """
    with open(input_file, "r") as fr1:
        gold_examples = fr1.read().split("\n\n")
    model_inputs = []
    for exg in gold_examples:
        input, output = exg.split("Output:")
        if input[-1] == "\n":
            input = input[:-1]
        if input[0] == "\n":
            input = input[1:]
        if output[-1] == "\n":
            output = output[:-1]
        input = user_input_start + input + user_input_end
        output = assistant_input_start + "Output:" + output + user_input_end
        model_inputs.append(input + output)
    return model_inputs


def main():
    with open(args.input_file, "r") as fr:
        abstract_id2tokenized_text = json.load(fr)

    # Read in predefined entity definitions and gold examples for in context learning.
    with open(args.definitions_file, "r") as fr:
        entity_definitions = fr.read()
    gold_examples = prepare_examples(args.examples_file)
    gold_examples = "\n\n".join(gold_examples)

    system_prompt_start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    system_prompt = system_prompt_start + PROMPT
    system_prompt = system_prompt + "\n\nHere are the definitions:\n" + entity_definitions + "<|eot_id|>" + \
                    "\n" + gold_examples

    user_input_start = "<|start_header_id|>user<|end_header_id|>\n\n"
    user_input_end = "<|eot_id|>\n"
    model_generation_start = "<|start_header_id|>assistant<|end_header_id|>"
    llama_pipeline, tokenizer = get_model(args.model_name)
    abs_id2model_output = {}
    for abs_id, sentences in abstract_id2tokenized_text.items():
        abs_id2model_output[abs_id] = []
        if "31761724" not in abs_id:
            continue
        for sent in sentences[:1]:
            query = "Input:\n" + sent["sent_text"] + user_input_end
            input_prompt = system_prompt + user_input_start
            input_prompt = input_prompt + query + model_generation_start
            print(input_prompt)
            print()
            output = run_llm(prompt=input_prompt, pipeline=llama_pipeline, tokenizer=tokenizer)
            abs_id2model_output[abs_id].append(output)

    with open(args.output_file, "w") as fw:
         json.dump(abs_id2model_output, fw)


def run_llm(prompt, pipeline, tokenizer):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=4096
    )
    generated_text = sequences[0]["generated_text"]
    response = generated_text[len(prompt):]
    return {"PROMPT": prompt, "MODEL_OUTPUT": response.strip()}


def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llama_pipeline = pipeline(
        "text-generation",
        model=model,
        model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True
        },
        tokenizer=tokenizer,
        device="cuda")
    return llama_pipeline, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct prompting using GPT family models."
    )
    parser.add_argument("--model_name", required=True, help="Which LLM to use")

    parser.add_argument("--definitions_file", type=str, default="definitions_all.txt",
                        help="TXT file contains the definition of entities.")
    parser.add_argument("--examples_file", type=str, default="examples_all.txt",
                        help="TXT file contains gold examples for in-context-learning.")

    parser.add_argument("--input_file", type=str, help="JSON file contains the input data.")
    parser.add_argument("--output_file", type=str, help="Name of the output file.")
    args = parser.parse_args()
    #model_id = args[1]
    main()



