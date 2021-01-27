import pandas as pd
import numpy
import re
from collections import defaultdict
import collections
from data.definition import LIVER_FINDINGS
from data.definition import TEMPLE_RADIO_1_2_SENTENCE_EMBEDDINGS, TEMPLE_RADIO_1_2_SENTENCES, DUPL_SENTENCES

from sentence_transformers import SentenceTransformer

def generate_clinical_bert_representation():
    # Use huggingface/transformers pre-trained model Bio_ClinicalBERT for mapping tokens to embeddings
    model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

    first_sentences = pd.read_csv(LIVER_FINDINGS, header=0)["Sentence 1"]
    first_sentences = list(first_sentences)

    sec_sentences = pd.read_csv(LIVER_FINDINGS, header=0)["Sentence 2"]
    sec_sentences = list(sec_sentences)
    sec_sentences = [sent for sent in sec_sentences if not (sent == '' or pd.isnull(sent))]

    first_sentences += sec_sentences
    sentences = [sent if sent.endswith('.') else sent + '.' for sent in first_sentences]

    dupl_sentences = [item for item, count in collections.Counter(sentences).items() if count > 1]
    pd.DataFrame(dupl_sentences).to_csv(DUPL_SENTENCES, index=False, header=False)

    unique_list = (list(set(sentences)))
    pd.DataFrame(unique_list).to_csv(TEMPLE_RADIO_1_2_SENTENCES, index=False, header=False)

    uniq_sentences = [sent.lower() for sent in unique_list]
    sentence_embeddings = model.encode(uniq_sentences)

    numpy.savetxt(TEMPLE_RADIO_1_2_SENTENCE_EMBEDDINGS, sentence_embeddings, delimiter=",")
    

if __name__ == '__main__':
    generate_clinical_bert_representation()