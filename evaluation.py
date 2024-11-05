import json
import os, re
import collections
import csv
import argparse


ENTITY_TYPES = ["diagnosis", "age_category", "genetic_effect", "model_type", "molecular_char", 
                "biomarker", "treatment", "response_to_treatment", "sample_type", "tumour_type", 
                "cancer_grade", "cancer_stage", "clinical_trial", "host_strain", "model_id"]
                # "gene_mutation_status"]


def read_in_gold(input_dir):
    """
    Read in gold annotations from a list of json files,
    preprocess them into a format used for comparison with model generations.
    We use the start and end offsets of an entity to identify that entity.
    """
    abstract_id2entities = {}
    for fl in os.listdir(input_dir):
        abstract_id = re.search(r"PMID\_\d+", fl, flags=0).group()
        with open(os.path.join(input_dir, fl), "r") as fr:
            annotations = json.load(fr)
        abstract_id2entities[abstract_id] = annotations

    gold4eval = {}
    for abs_id, annotations in abstract_id2entities.items():
        offset2anno = {}
        for anno in annotations:
            # Gene_mutation_status tag has been changed to
            # genetic_effect in the annotation guidelines
            if anno["tags"][0] == "gene_mutation_status":
                anno["tags"] = ["genetic_effect"]
            # This is the relation tag, ignore for right now
            if "from" in anno and "to" in anno:
                continue
            if "qualifier of" in anno["tags"]:
                continue
            offset2anno[(anno["startOffset"], anno["endOffset"])] = anno
        gold4eval[abs_id] = offset2anno
    return gold4eval


def read_in_pred(input_file):
    """
    We use the start and end offsets of an entity to identify that entity.
    """
    with open(input_file, "r") as fr:
        model_generation = json.load(fr)

    generation4eval = {}
    for abs_id, generations in model_generation.items():
        offset2gen = {}
        for gen in generations:
            offset2gen[(gen["startOffset"], gen["endOffset"])] = gen
        generation4eval[abs_id] = offset2gen
    return generation4eval


def compare(gold_file, pred_file):
    gold4eval = read_in_gold(gold_file)
    pred4eval = read_in_pred(pred_file)

    # This gold might contain all the annotations, including the ones in training set,
    # only keep the ones are in the same split as the prediction
    gold4eval = {k: v for k, v in gold4eval.items() if k in pred4eval}

    true_pos, false_pos, false_neg = 0, 0, 0

    # tp: true_positive; fn: false_negative; fp: false_positive
    tp_per_label = collections.defaultdict(int)
    fn_per_label = collections.defaultdict(int)
    fp_per_label = collections.defaultdict(int)

    all_f_scores = []
    for abs_id, gold_anno in gold4eval.items():
        # if "30395907" not in abs_id:
        #     continue
        print(abs_id)
        pred_gen = pred4eval[abs_id]

        # Group entities by their entity types
        gold_group_by_label = {ent: [] for ent in ENTITY_TYPES}
        pred_group_by_label = {ent: [] for ent in ENTITY_TYPES}

        all_gold = []
        all_pred = []
        for offset, gold_ent in gold_anno.items():
            all_gold.append([offset[0], offset[1], gold_ent["tags"][0], gold_ent["textProvided"]])
            gold_group_by_label[gold_ent["tags"][0]].append([offset[0], offset[1], gold_ent["textProvided"]])
        for p_offset, pred_ent in pred_gen.items():
            if p_offset[0] is False or p_offset[1] is False or pred_ent["tags"][0] is False:
                # The pred entities might not be well formatted.
                continue
            all_pred.append([p_offset[0], p_offset[1], pred_ent["tags"][0], pred_ent["textProvided"]])
            pred_group_by_label[pred_ent["tags"][0]].append([p_offset[0], p_offset[1], pred_ent["textProvided"]])

        for ent_type in ENTITY_TYPES:
            # print(ent_type, len(gold_group_by_label[ent_type]), len(pred_group_by_label[ent_type]))
            e_tp = [ele for ele in gold_group_by_label[ent_type] if ele in pred_group_by_label[ent_type]]
            # False negatives: the ones in gold, but not in pred.
            e_fn = [ele for ele in gold_group_by_label[ent_type] if ele not in pred_group_by_label[ent_type]]
            # False positives: the ones in pred, but not in gold.
            e_fp = [ele for ele in pred_group_by_label[ent_type] if ele not in gold_group_by_label[ent_type]]

            tp_per_label[ent_type] += len(e_tp)
            fn_per_label[ent_type] += len(e_fn)
            fp_per_label[ent_type] += len(e_fp)

        # After enumerating all the entities in an abstract, add up the tp, fn, fp for all entity types
        tp = [ele for ele in all_gold if ele in all_pred]
        fn = [ele for ele in all_gold if ele not in all_pred]
        fp = [ele for ele in all_pred if ele not in all_gold]

        true_pos += len(tp)
        false_neg += len(fn)
        false_pos += len(fp)

    # After get the tp, fn, fp for all abstracts, calculate the scores
    for ent_type in ENTITY_TYPES:
        # Compute for a single entity type
        ent_tp = tp_per_label[ent_type]
        ent_fn = fn_per_label[ent_type]
        ent_fp = fp_per_label[ent_type]
        # print(ent_type, ent_tp, ent_fn, ent_fp)
        # print(ent_type, all_tp, all_fn, all_fp)
        ent_precision = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) else 0
        ent_recall = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) else 0
        ent_f = 2 * (ent_precision * ent_recall) / (ent_precision + ent_recall) if (ent_precision + ent_recall) else 0
        all_f_scores.append(ent_f)
        print(ent_type, ent_tp, "ent_fn: ", ent_fn, "ent_fp: ", ent_fp)
        print("precision: ", ent_precision, "recall: ", ent_recall, "f: ", ent_f)
        print()

    # Compute overall scores across all entity types (micro f)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0
    f_score = 2 * (precision * recall) / (precision + recall)
    print("Results for micro F across all abstracts")
    print(true_pos, "false_neg", false_neg, "false_pos", false_pos)
    print(precision, recall, f_score)

    # # Compute average scores across all entity types (macro f)
    #
    # print("Results for macro F across all entity types")
    # print(sum(all_f_scores)/len(all_f_scores))


def compare_overlap(gold_file, pred_file):
    gold4eval = read_in_gold(gold_file)
    pred4eval = read_in_pred(pred_file)

    # This gold might contain all the annotations, including the ones in training set,
    # only keep the ones are in the same split as the prediction
    gold4eval = {k: v for k, v in gold4eval.items() if k in pred4eval}

    true_pos, false_pos, false_neg = 0, 0, 0
    tp_per_label = collections.defaultdict(int)
    fn_per_label = collections.defaultdict(int)
    fp_per_label = collections.defaultdict(int)
    all_f_scores = []
    for abs_id, gold_anno in gold4eval.items():
        # if "30395907" not in abs:
        #     continue
        print(abs_id)
        gold_group_by_label = {ent: [] for ent in ENTITY_TYPES}
        pred_group_by_label = {ent: [] for ent in ENTITY_TYPES}
        pred_gen = pred4eval[abs_id]

        all_gold = []
        all_pred = []
        for offset, gold_ent in gold_anno.items():
            all_gold.append([offset[0], offset[1], gold_ent["tags"][0]])
            gold_group_by_label[gold_ent["tags"][0]].append([offset[0], offset[1]])#, gold_ent["textProvided"]
        for p_offset, pred_ent in pred_gen.items():
            if p_offset[0] is False or p_offset[1] is False or pred_ent["tags"][0] is False:
                continue
            all_pred.append([p_offset[0], p_offset[1], pred_ent["tags"][0]])
            pred_group_by_label[pred_ent["tags"][0]].append([p_offset[0], p_offset[1]])

        for ent_type in ENTITY_TYPES:
            ent_gold = gold_group_by_label[ent_type]
            ent_pred = pred_group_by_label[ent_type]

            for g_item in ent_gold:
                has_in_pred = False
                for p_item in ent_pred:
                    # Compare if g_item overlaps p_item by comparing their start, end offsets
                    over_lapping = is_overlap(int(g_item[0]), int(g_item[1]), p_item[0], p_item[1])
                    if over_lapping:
                        tp_per_label[ent_type] += 1
                        has_in_pred = True
                if not has_in_pred:
                    fn_per_label[ent_type] += 1
            for p_item in ent_pred:
                has_in_gold = False
                for g_item in ent_gold:
                    over_lapping = is_overlap(int(g_item[0]), int(g_item[1]), p_item[0], p_item[1])
                    if over_lapping:
                        has_in_gold = True
                if not has_in_gold:
                    fp_per_label[ent_type] += 1
        #     print(tp_per_label[ent_type], fn_per_label[ent_type], fp_per_label[ent_type])
        #     print()
        # print()
    for ent_type in ENTITY_TYPES:
        ent_tp = tp_per_label[ent_type]
        ent_fn = fn_per_label[ent_type]
        ent_fp = fp_per_label[ent_type]
        ent_precision = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) else 0
        ent_recall = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) else 0
        ent_f = 2 * (ent_precision * ent_recall) / (ent_precision + ent_recall) if (
                ent_precision + ent_recall) else 0
        all_f_scores.append(ent_f)
        print(ent_type, ent_tp, "ent_fn: ", ent_fn, "ent_fp: ", ent_fp)
        print("precision:", ent_precision, "recall", ent_recall, "f", ent_f)
        print()

        true_pos += tp_per_label[ent_type]
        false_neg += fn_per_label[ent_type]
        false_pos += fp_per_label[ent_type]

    all_prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    all_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0
    f_score = 2 * (all_prec * all_recall) / (all_prec + all_recall)
    print(true_pos, "false_neg: ", false_neg, "false_pos: ", false_pos)
    print(true_pos, "false_neg", false_neg, "false_pos", false_pos)
    print(all_prec, all_recall, f_score)

    print("Average across all entity types:")
    print(sum(all_f_scores)/len(all_f_scores))


def is_overlap(start_g, end_g, start_p, end_p):
    # print(start_g, end_g, start_p, end_p)
    # print(type(start_g), type(end_g), type(start_p), type(end_p))
    if start_g <= start_p <= end_g:
        return True
    if start_g <= end_p <= end_g:
        return True
    if start_p <= start_g <= end_p:
        return True
    if start_p <= end_g <= end_p:
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate predicted output against gold annotations"
    )
    parser.add_argument(
        "--gold_dir", help="Gold annotations, a list of json files",
        default="merged_annotated_data/adjudicated_gold"
    )
    parser.add_argument(
        "--pred_file", help="A predicted output json file"
    )
    
    parser.add_argument("--exact_match", action="store_true", help="do exact match eval")
    args = parser.parse_args()

    if args.exact_match:
        compare(args.gold_dir, args.pred_file)
    else:
        compare_overlap(args.gold_dir, args.pred_file)
