import openai
import json
import argparse

# This is the overall instruction describing the task.
PROMPT = "You will be given a sentence from a paper on PDCM. Please extract entities as defined below and return as an XML." \
         " In the XML, please mark the start and end of the entity with <entity_type></entity_type>. " \
         "Please return one entity at one time. That is, if there are n entities in the sentence, " \
         "print the sentence with the marked entity for n times. Do not change anything else in the sentence."


def run_model(client, model_name, instruction, input):
    # For GPT-4o, the prompt generated before becomes the "instruction" here,
    # the sentence to extract entities from becomes the "input".
    response = client.chat.completions.create(
      model=model_name,
      messages=[
        {
          "role": "system",
          "content": instruction
        },
        {
          "role": "user",
          "content": input
        }
      ],
      temperature=1
    )
    return response.choices[0].message.content


def get_prompt(prompt, definitions, examples):
    # Concatenate overall_instruction, entity definitions, gold examples, this whole thing becomes the prompt to LLM
    input_to_llm = prompt + "\n\nHere are the definitions:\n" + definitions + "\n" + examples + "\n"
    return input_to_llm


def main():
    with open(args.input_file, "r") as fr:
        abstract_id2tokenized_text = json.load(fr)

    # Read in predefined entity definitions and gold examples for in context learning.
    with open(args.definitions_file, "r") as fr:
        entity_definitions = fr.read()
    with open(args.examples_file, "r") as fr1:
        gold_examples = fr1.read()

    abs_id2model_output = {}
    client = openai.OpenAI(api_key=args.api_key)
    for abs_id, sentences in abstract_id2tokenized_text.items():
        # if "31761724" not in abs_id:
        #     continue
        abs_id2model_output[abs_id] = []
        for sent in sentences:
            query = "Input:\n" + sent["sent_text"]
            # Get prompt
            prompt = get_prompt(PROMPT, entity_definitions, gold_examples)
            output = run_model(client=client,
                               model_name=args.model_name,
                               instruction=prompt,
                               input=query)
            # The output will be in the same format as the OUTPUT in examples_all.txt, parse it to the format you want.
            abs_id2model_output[abs_id].append(output)

    # Write LLM generation to a json file.
    with open(args.output_file, "w") as fw:
        json.dump(abs_id2model_output, fw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct prompting using GPT family models."
    )
    parser.add_argument("--model_name", required=True, help="Which LLM to use")
    parser.add_argument("--api_key", required=True, help="OPENAI API key")

    parser.add_argument("--definitions_file", type=str, default="definitions_all.txt",
                        help="TXT file contains the definition of entities.")
    parser.add_argument("--examples_file", type=str, default="examples_all.txt",
                        help="TXT file contains gold examples for in-context-learning.")

    parser.add_argument("--input_file", type=str, help="JSON file contains the input data.")
    parser.add_argument("--output_file", type=str, help="Name of the output file.")

    args = parser.parse_args()

    main()
