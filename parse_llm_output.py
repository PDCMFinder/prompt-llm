import json
import re
import os
import argparse


def read_in(input_file):
    with open(input_file, "r") as fr:
        input_data = json.load(fr)
    return input_data


def read_in_abstracts(input_dir):
    abstract_id2text = {}
    for fl in os.listdir(input_dir):
        #abstract_id = re.search(r"PMID\_\d+", fl, flags=0).group()
        abstract_id = re.search(r"PMC\d+", fl, flags=0).group()
        with open(os.path.join(input_dir, fl), "r") as fr:
            text = fr.read()
        abstract_id2text[abstract_id] = text
    return abstract_id2text


def extract_answer(generated):
    num_ent = re.findall(r"</", generated, flags=0)
    if len(num_ent) > 1:
        print(f"More than one entity in a line: {num_ent} {generated}\n")
    s = generated.find(">") + 1
    e = generated.find("</")
    if "</diagnosis" in generated:
        ent_type = "diagnosis"
        ent_type_mark = re.search(r"<diagnosis\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</age_category" in generated:
        ent_type = "age_category"
        ent_type_mark = re.search(r"<age_category\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</genetic_effect" in generated:
        ent_type = "genetic_effect"
        ent_type_mark = re.search(r"<genetic_effect\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</model_type" in generated:
        ent_type = "model_type"
        ent_type_mark = re.search(r"<model_type\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</biomarker" in generated:
        ent_type = "biomarker"
        ent_type_mark = re.search(r"<biomarker\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</treatment" in generated:
        ent_type = "treatment"
        ent_type_mark = re.search(r"<treatment\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</molecular_char" in generated:
        ent_type = "molecular_char"
        # print(generated)

        ent_type_mark = re.search(r"<molecular_char\w*>", generated, flags=0)
        if not ent_type_mark:
            ent_type_mark = re.search(r"\(molecular_char\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</response_to_treatment" in generated:
        ent_type = "response_to_treatment"
        ent_type_mark = re.search(r"<response_to_treatment\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</sample_type" in generated:
        ent_type = "sample_type"
        ent_type_mark = re.search(r"<sample_type\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</tumour_type" in generated:
        ent_type = "tumour_type"
        ent_type_mark = re.search(r"<tumour_type\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</cancer_grade" in generated:
        ent_type = "cancer_grade"
        ent_type_mark = re.search(r"<cancer_grade\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</cancer_stage" in generated:
        ent_type = "cancer_stage"
        ent_type_mark = re.search(r"<cancer_stage\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</clinical_trial" in generated:
        ent_type = "clinical_trial"
        ent_type_mark = re.search(r"<clinical_trial\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</host_strain" in generated:
        ent_type = "host_strain"
        ent_type_mark = re.search(r"<host_strain\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    elif "</model_id" in generated:
        ent_type = "model_id"
        ent_type_mark = re.search(r"<model_id\w*>", generated, flags=0)
        offset = len(ent_type_mark.group())
    else:
        return False
    assert s < e
    orig_s = s - offset
    orig_e = e - offset
    return [s, e, ent_type, orig_s, orig_e]


def parse(input_file, info_map, original_data_path):
    llm_generations = read_in(input_file)
    offset_info = read_in(info_map)
    abstract_id2text = read_in_abstracts(original_data_path)
    parsed_generations = {}
    for abstract_id, extractions in llm_generations.items():
        abs_info = offset_info[abstract_id]
        abs_text = abstract_id2text[abstract_id]

        #if "32294323" not in abstract_id:
        #    continue
        #print(f'Extractions: {extractions}\n')
        parsed_generations[abstract_id] = []
        for s_idx, sent in enumerate(extractions):
            if isinstance(sent, dict):
                sent = sent["MODEL_OUTPUT"]
            #print(f'Before check output: {sent}\n')
            if "\n\n(1)" not in sent and "\n(1)" not in sent:
                if "no entities" in sent.lower() or "0 entities" in sent.lower():
                    continue
                if "1 entity" not in sent.lower():
                    # print("line122", repr(sent))
                    continue
            #print(f'Llama: {repr(sent)}\n')
            if "\n\n(1)" in sent:
                marked_sents = "(1)" + sent.split("\n\n(1)")[-1]
            else:
                marked_sents = "(1)" + sent.split("\n(1)")[-1]

            marked_sents = marked_sents.split("\n")
            sent_char_info = abs_info[s_idx]
            #print(f'Splitted marked sentence: {marked_sents}\n')
            for item in marked_sents:

                if len(item) == 0:
                    continue
                idx = re.match("\\(\\d+\\)", item, flags=0)
                #print(f'IDX: {idx}\n')
                if not idx:
                    continue
                #print(f'idx: {idx}')
                sent_wt_mark = item[idx.span()[-1]:]
                if sent_wt_mark[0] == " ":
                    sent_wt_mark = sent_wt_mark[1:]
                #print(f'sent_mark: {sent_wt_mark}\n')
                extracted = extract_answer(sent_wt_mark)
                if not extracted:
                    continue
                #print(f'Extracted: s,e,et, os, oe {extracted}\n')
                ent_start, ent_end, ent_type, orig_s, orig_e = extracted
                #print(f'entity type: {ent_type}')
                if ent_type is None:
                    continue
                

                marked_sent_ent_text = sent_wt_mark[ent_start: ent_end]
                ent_doc_level_s = orig_s + sent_char_info["char_start"]
                ent_doc_level_e = orig_e + sent_char_info["char_start"]
                doc_ent_text = abs_text[ent_doc_level_s: ent_doc_level_e]
                # This is because the output is not exactly the same as original sentence
                if marked_sent_ent_text != doc_ent_text:
                    print(f'\nPMID: {abstract_id}\n')
                    #print(f'Abstract: {abs_text}\n')
                    #print(f'Marked sentence: {marked_sents}\n')
                    print(f'Item: {item}')
                    print(f'Doc entity text: {doc_ent_text}; Marked sentence entity text: {marked_sent_ent_text}') 
                    print('########## not the same ###########\n\n')
                    continue
                one_extracted_ent = {"startOffset": ent_doc_level_s,
                                     "endOffset": ent_doc_level_e,
                                     "tags": [ent_type],
                                     "id": abstract_id + "_" + str(ent_doc_level_s),
                                     "textProvided": doc_ent_text}
                parsed_generations[abstract_id].append(one_extracted_ent)
    print(f'\n\nParsed generations: {parsed_generations}')
    return parsed_generations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the output of LLMs."
    )
    parser.add_argument("--llm_generation", type=str, help="JSON file contains the generations of LLMs.")
    parser.add_argument("--parsed_output", type=str, help="Name of the output file.")
    parser.add_argument("--information_map", type=str,
                        help="Keep track of the offsets and other info in the original abstract.")
    parser.add_argument("--abstract_text_dir", type=str, help="Path to abstract texts, a list of txt files.")

    args = parser.parse_args()

    parsed_generations = parse(input_file=args.llm_generation,
                               info_map=args.information_map,
                               original_data_path=args.abstract_text_dir)

    with open(args.parsed_output, "w") as fw:
        json.dump(parsed_generations, fw)


