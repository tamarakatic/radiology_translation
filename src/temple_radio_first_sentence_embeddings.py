import pandas as pd
import numpy
import re
from collections import defaultdict
from data.definition import LIVER_FINDINGS
from data.definition import TEMPLE_RADIO_SENTENCE_EMBEDDINGS, TEMPLE_RADIO_SENTENCES

from sentence_transformers import SentenceTransformer

def generate_clinical_bert_representation():
    # Use huggingface/transformers pre-trained model Bio_ClinicalBERT for mapping tokens to embeddings
    model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

    sentences = pd.read_csv(LIVER_FINDINGS, header=0)["Sentence 1"]
    sentences = list(sentences)
    pd.DataFrame(sentences).to_csv(TEMPLE_RADIO_SENTENCES, index=False, header=False)

    # unique_list = (list(set(sentences)))
    sentences = [sent.lower() for sent in sentences]

    sentence_embeddings = model.encode(sentences)

    numpy.savetxt(TEMPLE_RADIO_SENTENCE_EMBEDDINGS, sentence_embeddings, delimiter=",")
    

if __name__ == '__main__':
    generate_clinical_bert_representation()