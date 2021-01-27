import pandas as pd
import numpy as np
import re
from collections import defaultdict
from data.definition import LIVER_FINDINGS
from data.definition import TEMPLE_RADIO_SECOND_SENTENCE_EMBEDDINGS, TEMPLE_RADIO_SECOND_SENTENCES

from sentence_transformers import SentenceTransformer

def generate_clinical_bert_representation():
    # Use huggingface/transformers pre-trained model Bio_ClinicalBERT for mapping tokens to embeddings
    model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

    sentences = pd.read_csv(LIVER_FINDINGS, header=0)["Sentence 2"]
    sentences = list(sentences)
    sentences = [sent for sent in sentences if not (sent == '' or pd.isnull(sent))]
    
    pd.DataFrame(sentences).to_csv(TEMPLE_RADIO_SECOND_SENTENCES, index=False, header=False)

    # unique_list = (list(set(sentences)))
    sentences = [sent.lower() for sent in sentences]

    sentence_embeddings = model.encode(sentences)

    np.savetxt(TEMPLE_RADIO_SECOND_SENTENCE_EMBEDDINGS, sentence_embeddings, delimiter=",")
    

if __name__ == '__main__':
    generate_clinical_bert_representation()