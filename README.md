# prompt_pdcr_ebi


1. Data pre-processing
    1. Split an abstract into a list of sentences, keep track of the start and end of character offsets of the sentence.
    2. The idea is, we will send one sentence to the LLM each time to extract entities, then assemble all the entities in this abstract after we go through every sentence.
    

```python
def read_in_abstracts(input_dir):
    abstract_id2text = {}
    for fl in os.listdir(input_dir):
        abstract_id = re.search(r"PMID\_\d+", fl, flags=0).group()
        with open(os.path.join(input_dir, fl), "r") as fr:
            text = fr.read()
        abstract_id2text[abstract_id] = text
    return abstract_id2text

def abstract_to_sent_list(input_dir, tokenization_path):
    abstract_id2_text = read_in_abstracts(input_dir)
    abstract_id2tokenized_text = {}
    for abs_id, abstract_text in abstract_id2_text.items():
        # abs_id: PMID_30859564
        tokenized_id = abs_id + "_tokenSpans.txt"
        tokenized = read_sent_boundary(os.path.join(tokenization_path, tokenized_id))
        to_verify = ""
        tokenized_abs = []
        for s_id, tokenize_info in enumerate(tokenized):
            _, s_start, s_end = tokenize_info
            one_sent = abstract_text[int(s_start): int(s_end)]
            to_verify += one_sent
            to_verify += " "
            tokenized_abs.append(
                {"sent_idx": s_id, "char_start": int(s_start), "char_end": int(s_end),
                 "sent_text": one_sent})
        to_verify = to_verify[:-1]
        abstract_id2tokenized_text[abs_id] = tokenized_abs
        assert to_verify == abstract_text
```

2. Generate input for LLMs
    1. what you need: 
        1. definitions of all entities, e.g. “definitions_all.txt”
            
            ```
            **diagnosis**: Diagnosis at the time of collection of the patient tumour used in the cancer model. Some examples are Colorectal carcinoma, triple-negative breast cancer, TMBC, Ewing sarcoma, EWS.
            **age_category**: Age category of the patient at the time of tissue sampling. Some examples are adult, pediatric, child, fetus, fetal, young adult.
            ```
            
        2. gold examples for LLMs to learn from (in-context-learning), e.g. “examples_all.txt”
            
            ```
            Input: 
            There were 13 missense mutations identified in the xenograft that were not present in the patient's primary tumor and there were no new nonsense mutations.
            Output:
            The entities in this sentence are:
            {"genetic_effect": [missense mutations], "model_type": [xenograft], "tumour_type": [primary]}
            There are 3 entities in totoal. Now I will print out the sentence for 3 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
            (1) There were 13 <genetic_effect>missense mutations</genetic_effect> identified in the xenograft that were not present in the patient's primary tumor and there were no new nonsense mutations.
            (2) There were 13 missense mutations identified in the <model_type>xenograft</model_type> that were not present in the patient's primary tumor and there were no new nonsense mutations.
            (3) There were 13 missense mutations identified in the xenograft that were not present in the patient's <tumour_type>primary</tumour_type> tumor and there were no new nonsense mutations.
            ```
            
    2. input format:
        1. instruction: overall_instruction (describing the task) + definitions + gold examples 
        2. query/input: one sentence from the abstract, i.e., the sentence you want to extract entities from
    
    ```xml
    You will be given a sentence from a paper on PDCM. Please extract entities as defined below and return as an XML. In the XML, please mark the start and end of the entity with <entity_type></entity_type>. Please return one entity at one time. That is, if there are n entities in the sentence, print the sentence with the marked entity for n times. Do not change anything else in the sentence.
    
    Here are the definitions:
    diagnosis: Diagnosis at the time of collection of the patient tumour used in the cancer model. Some examples are Colorectal carcinoma, triple-negative breast cancer, TMBC, Ewing sarcoma, EWS.
    age_category: Age category of the patient at the time of tissue sampling. Some examples are adult, pediatric, child, fetus, fetal, young adult.
    genetic_effect: Any form of chromosomal rearrangement or gene-level changes. Some examples are missense, deletion, amplification, gene rearrangement, fusions, de novo gene fusions, deleterious mutations, missense mutations, deficient, activation, R1181C, Exon 2-7 deletion, microsatellite instability, homologous recombination deficiency, HRD, defect in HR, alterations, V600E.
    model_type: Type of patient-derived model. Some examples are xenograft, organoid, tumour organoid, 3D culture, 3D cell line, 2D cell line, tumouroid, spheroid, cell line, PDX, PDO, PDC, patient-derived xenograft.
    molecular_char: Data or assay generated from or performed on the model in this study. Some examples are RNA sequencing, WES, whole-some sequencing, RT-PCR, immunohistochemistry analysis, western blot analysis, mRNA expression, microarray analysis, FISH, methylation assays, mass spectrometry, SNP array.
    biomarker: Gene, protein or other substance that has been tested to reveal important details about patient’s cancer/disease state. Some examples are BRCA1 mutation, IDH mutant, CD44, TP53, BRAF v600e.
    treatment: Treatment received by the patient or tested on the model. Some examples are surgery, mastectomy, lymphadenectomy, radiation therapy, chemotherapy, immunotherapy, targeted therapy, FOLFOX, cisplatin, FEC regimen, antibody, alkylating agent, EGRF inhibitor.
    response_to_treatment: Effect of the treatment on the patient's tumour or model. Some examples are CR (complete response), PR (partial response), PD (progressive disease), SD (stable disease), progression-free survival, reduces tumour growth, prevents tumour relapse, sustained regression.
    sample_type: The type of material used to generate the model or how this material was obtained. Some examples are tissue fragments, cell suspension, cells, biopsy, ascites, surgical specimen, autopsy.
    tumour_type: collected tumour type used for generating the model. Some examples are primary, metastatic, recurrent, refractory, pre-malignant, pre-cancerous, treatment refractory, treatment-resistant, endocrine resistant.
    cancer_grade: Quantitive or qualitative grade reflecting how quickly the cancer is likely to grow. Some examples are Grade 1, Grade 2, Grade 3, Grade 4, low-grade, high-grade.
    cancer_stage: Information about the cancer’s extent in the body according to specific type of cancer staging system. Some examples are pT4N1M0 (TNM system - Tumour size, Node involvement, Metastasis), Stage IV (number system).
    clinical_trial: The type of clinical trial or Clinicaltrials.org identifier. Some examples are NCT03668431, phase II, 1x1x1 experimental design, PDX clinical trial, PCT.
    host_strain: The name of the mouse host strain where the tissue sample was engrafted for generating the PDX model. One example is NOD-SCID.
    model_id: ID of the patient-derived cancer model generated in this study. One example is PHLC402.
    
    Input: 
    There were 13 missense mutations identified in the xenograft that were not present in the patient's primary tumor and there were no new nonsense mutations.
    Output:
    The entities in this sentence are:
    {"genetic_effect": [missense mutations], "model_type": [xenograft], "tumour_type": [primary]}
    There are 3 entities in totoal. Now I will print out the sentence for 3 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
    (1) There were 13 <genetic_effect>missense mutations</genetic_effect> identified in the xenograft that were not present in the patient's primary tumor and there were no new nonsense mutations.
    (2) There were 13 missense mutations identified in the <model_type>xenograft</model_type> that were not present in the patient's primary tumor and there were no new nonsense mutations.
    (3) There were 13 missense mutations identified in the xenograft that were not present in the patient's <tumour_type>primary</tumour_type> tumor and there were no new nonsense mutations.
    
    Input: 
    Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
    Output:
    The entities in this sentence are:
    {"molecular_char": [molecular, cellular, genetic, epigenetic characterization], "model_type": [orthotopic xenograft], "cancer_stage": [stage 4], "diagnosis": [neuroblastoma]}
    There are 7 entities in totoal. Now I will print out the sentence for 7 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
    (1) Here we present the detailed <molecular_char>molecular<molecular_char>, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
    (2) Here we present the detailed molecular, <molecular_char>cellular</molecular_char>, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
    (3) Here we present the detailed molecular, cellular, <molecular_char>genetic</molecular_char> and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
    (4) Here we present the detailed molecular, cellular, genetic and <molecular_char>epigenetic characterization</molecular_char> of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
    (5) Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an <model_type>orthotopic xenograft</model_type> derived from a high-risk stage 4 neuroblastoma patient.
    (6) Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk <cancer_stage>stage 4</cancer_stage> neuroblastoma patient.
    (7) Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 <diagnosis>neuroblastoma</diagnosis> patient.
    
    Input: 
    Neuroblastoma is a pediatric cancer of the developing sympathoadrenal lineage. 
    Output:
    The entities in this sentence are:
    {"diagnosis": [Neuroblastoma], "age_category": [pediatric]}
    There are 2 entities in totoal. Now I will print out the sentence for 2 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
    (1) <diagnosis>Neuroblastoma</diagnosis> is a pediatric cancer of the developing sympathoadrenal lineage. 
    (2) Neuroblastoma is a <age_category>pediatric</age_category> cancer of the developing sympathoadrenal lineage. 
    
    Input: 
    With these PDXs, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (PDX clinical trial or PCT) to assess the population responses to 62 treatments across six indications.
    Output:
    The entities in this sentence are:
    {"model_type": [PDXs], "clinical_trial": [PDX clinical trial, PCT]}
    There are 3 entities in totoal. Now I will print out the sentence for 3 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
    (1) With these <model_type>PDXs<model_type>, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (PDX clinical trial or PCT) to assess the population responses to 62 treatments across six indications.
    (2) With these PDXs, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (<clinical_trial>PDX clinical trial</clinical_trial> or PCT) to assess the population responses to 62 treatments across six indications.
    (3) With these PDXs, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (PDX clinical trial or <clinical_trial>PCT<clinical_trial>) to assess the population responses to 62 treatments across six indications.
    
    Input: 
    As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
    Output:
    The entities in this sentence are:
    {"sample_type": [autopsy], "biomarker": [ALK], "model_type": [PDX], "treatment": [pan-kinase inhibitor, lestaurtinib], "response_to_treatment": [decrease in tumor growth]}
    There are 6 entities in totoal. Now I will print out the sentence for 6 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
    (1) As <sample_type>autopsy</sample_type> specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
    (2) As autopsy specimens had an <biomarker>ALK<biomarker> R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
    (3) As autopsy specimens had an ALK R1181C mutation, <model_type>PDX</model_type> tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
    (4) As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the <treatment>pan-kinase inhibitor</treatment> lestaurtinib but demonstrated no decrease in tumor growth.
    (5) As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor <treatment>lestaurtinib</treatment> but demonstrated no decrease in tumor growth.
    (6) As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no <response_to_treatment>decrease in tumor growth</response_to_treatment>.
    
    Input:
    Here we generate a living¬†organoid biobank from patients with locally advanced rectal cancer (LARC) treated with neoadjuvant chemoradiation (NACR) enrolled in a phase III clinical trial.
    ```
    
    basically, for each sentence, send the whole thing as above to the model
    

3. Run GPT-4o
    1. API-Key
    2. run model
    
4. Post processing
    
    The model will generate output looks like this:
    

```xml
The entities in this sentence are:
{"diagnosis": [Neuroblastoma], "age_category": [pediatric]}
There are 2 entities in totoal. Now I will print out the sentence for 2 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
(1) <diagnosis>Neuroblastoma</diagnosis> is a pediatric cancer of the developing sympathoadrenal lineage. 
(2) Neuroblastoma is a <age_category>pediatric</age_category> cancer of the developing sympathoadrenal lineage. 
```

To use it, you can use Regular expression to extract the entities and entity type. To get their character level offsets in the original abstract, first get their sentence level character offsets, then compute its abstract level offsets.

e.g. `gpt_4o_output_dev.json` and `gpt_4o_output_dev_parsed.json`

5. Evaluation

evaluation.py

---

6. Get input for LLAMA3.1 models

[https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)

```xml
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You will be given a sentence from a paper on PDCM. Please extract entities as defined below and return as an XML. In the XML, please mark the start and end of the entity with <entity_type></entity_type>. Please return one entity at one time. That is, if there are n entities in the sentence, print the sentence with the marked entity for n times. Do not change anything else in the sentence.

Here are the definitions:
diagnosis: Diagnosis at the time of collection of the patient tumour used in the cancer model. Some examples are Colorectal carcinoma, triple-negative breast cancer, TMBC, Ewing sarcoma, EWS.
age_category: Age category of the patient at the time of tissue sampling. Some examples are adult, pediatric, child, fetus, fetal, young adult.
genetic_effect: Any form of chromosomal rearrangement or gene-level changes. Some examples are missense, deletion, amplification, gene rearrangement, fusions, de novo gene fusions, deleterious mutations, missense mutations, deficient, activation, R1181C, Exon 2-7 deletion, microsatellite instability, homologous recombination deficiency, HRD, defect in HR, alterations, V600E.
model_type: Type of patient-derived model. Some examples are xenograft, organoid, tumour organoid, 3D culture, 3D cell line, 2D cell line, tumouroid, spheroid, cell line, PDX, PDO, PDC, patient-derived xenograft.
molecular_char: Data or assay generated from or performed on the model in this study. Some examples are RNA sequencing, WES, whole-some sequencing, RT-PCR, immunohistochemistry analysis, western blot analysis, mRNA expression, microarray analysis, FISH, methylation assays, mass spectrometry, SNP array.
biomarker: Gene, protein or other substance that has been tested to reveal important details about patient’s cancer/disease state. Some examples are BRCA1 mutation, IDH mutant, CD44, TP53, BRAF v600e.
treatment: Treatment received by the patient or tested on the model. Some examples are surgery, mastectomy, lymphadenectomy, radiation therapy, chemotherapy, immunotherapy, targeted therapy, FOLFOX, cisplatin, FEC regimen, antibody, alkylating agent, EGRF inhibitor.
response_to_treatment: Effect of the treatment on the patient's tumour or model. Some examples are CR (complete response), PR (partial response), PD (progressive disease), SD (stable disease), progression-free survival, reduces tumour growth, prevents tumour relapse, sustained regression.
sample_type: The type of material used to generate the model or how this material was obtained. Some examples are tissue fragments, cell suspension, cells, biopsy, ascites, surgical specimen, autopsy.
tumour_type: collected tumour type used for generating the model. Some examples are primary, metastatic, recurrent, refractory, pre-malignant, pre-cancerous, treatment refractory, treatment-resistant, endocrine resistant.
cancer_grade: Quantitive or qualitative grade reflecting how quickly the cancer is likely to grow. Some examples are Grade 1, Grade 2, Grade 3, Grade 4, low-grade, high-grade.
cancer_stage: Information about the cancer’s extent in the body according to specific type of cancer staging system. Some examples are pT4N1M0 (TNM system - Tumour size, Node involvement, Metastasis), Stage IV (number system).
clinical_trial: The type of clinical trial or Clinicaltrials.org identifier. Some examples are NCT03668431, phase II, 1x1x1 experimental design, PDX clinical trial, PCT.
host_strain: The name of the mouse host strain where the tissue sample was engrafted for generating the PDX model. One example is NOD-SCID.
model_id: ID of the patient-derived cancer model generated in this study. One example is PHLC402.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Input: 
There were 13 missense mutations identified in the xenograft that were not present in the patient's primary tumor and there were no new nonsense mutations.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Output:
The entities in this sentence are:
{"genetic_effect": [missense mutations], "model_type": [xenograft], "tumour_type": [primary]}
There are 3 entities in totoal. Now I will print out the sentence for 3 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
(1) There were 13 <genetic_effect>missense mutations</genetic_effect> identified in the xenograft that were not present in the patient's primary tumor and there were no new nonsense mutations.
(2) There were 13 missense mutations identified in the <model_type>xenograft</model_type> that were not present in the patient's primary tumor and there were no new nonsense mutations.
(3) There were 13 missense mutations identified in the xenograft that were not present in the patient's <tumour_type>primary</tumour_type> tumor and there were no new nonsense mutations.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Input: 
Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Output:
The entities in this sentence are:
{"molecular_char": [molecular, cellular, genetic, epigenetic characterization], "model_type": [orthotopic xenograft], "cancer_stage": [stage 4], "diagnosis": [neuroblastoma]}
There are 7 entities in totoal. Now I will print out the sentence for 7 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
(1) Here we present the detailed <molecular_char>molecular<molecular_char>, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
(2) Here we present the detailed molecular, <molecular_char>cellular</molecular_char>, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
(3) Here we present the detailed molecular, cellular, <molecular_char>genetic</molecular_char> and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
(4) Here we present the detailed molecular, cellular, genetic and <molecular_char>epigenetic characterization</molecular_char> of an orthotopic xenograft derived from a high-risk stage 4 neuroblastoma patient.
(5) Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an <model_type>orthotopic xenograft</model_type> derived from a high-risk stage 4 neuroblastoma patient.
(6) Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk <cancer_stage>stage 4</cancer_stage> neuroblastoma patient.
(7) Here we present the detailed molecular, cellular, genetic and epigenetic characterization of an orthotopic xenograft derived from a high-risk stage 4 <diagnosis>neuroblastoma</diagnosis> patient.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Input: 
Neuroblastoma is a pediatric cancer of the developing sympathoadrenal lineage. <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Output:
The entities in this sentence are:
{"diagnosis": [Neuroblastoma], "age_category": [pediatric]}
There are 2 entities in totoal. Now I will print out the sentence for 2 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
(1) <diagnosis>Neuroblastoma</diagnosis> is a pediatric cancer of the developing sympathoadrenal lineage. 
(2) Neuroblastoma is a <age_category>pediatric</age_category> cancer of the developing sympathoadrenal lineage. <|eot_id|>

<|start_header_id|>user<|end_header_id|>

Input: 
With these PDXs, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (PDX clinical trial or PCT) to assess the population responses to 62 treatments across six indications.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Output:
The entities in this sentence are:
{"model_type": [PDXs], "clinical_trial": [PDX clinical trial, PCT]}
There are 3 entities in totoal. Now I will print out the sentence for 3 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
(1) With these <model_type>PDXs<model_type>, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (PDX clinical trial or PCT) to assess the population responses to 62 treatments across six indications.
(2) With these PDXs, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (<clinical_trial>PDX clinical trial</clinical_trial> or PCT) to assess the population responses to 62 treatments across six indications.
(3) With these PDXs, we performed in vivo compound screens using a 1 × 1 × 1 experimental design (PDX clinical trial or <clinical_trial>PCT<clinical_trial>) to assess the population responses to 62 treatments across six indications.<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Input: 
As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Output:
The entities in this sentence are:
{"sample_type": [autopsy], "biomarker": [ALK], "model_type": [PDX], "treatment": [pan-kinase inhibitor, lestaurtinib], "response_to_treatment": [decrease in tumor growth]}
There are 6 entities in totoal. Now I will print out the sentence for 6 times, each time with only one entity marked, the rest of the sentence will be exactly the same as the original sentence.
(1) As <sample_type>autopsy</sample_type> specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
(2) As autopsy specimens had an <biomarker>ALK<biomarker> R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
(3) As autopsy specimens had an ALK R1181C mutation, <model_type>PDX</model_type> tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no decrease in tumor growth.
(4) As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the <treatment>pan-kinase inhibitor</treatment> lestaurtinib but demonstrated no decrease in tumor growth.
(5) As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor <treatment>lestaurtinib</treatment> but demonstrated no decrease in tumor growth.
(6) As autopsy specimens had an ALK R1181C mutation, PDX tumor bearing animals were treated with the pan-kinase inhibitor lestaurtinib but demonstrated no <response_to_treatment>decrease in tumor growth</response_to_treatment>.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Input:
Accumulating evidence indicates that patient-derived organoids (PDOs) can predict drug responses in the clinic, but the ability of PDOs to predict responses to chemoradiation in cancer patients remains an open question.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```
