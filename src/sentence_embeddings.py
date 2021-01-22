import pandas as pd
import numpy
import re
from collections import defaultdict
from data.definition import FINDINGS_RADIOLOGY_SENTENCES_PER_REPORT
from data.definition import SENTENCE_EMBEDDINGS, SENTENCES

from sentence_transformers import SentenceTransformer

def generate_clinical_bert_representation():
    # Use huggingface/transformers pre-trained model Bio_ClinicalBERT for mapping tokens to embeddings
    model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

    sentences = pd.read_csv(FINDINGS_RADIOLOGY_SENTENCES_PER_REPORT, nrows=10000, header=0).first_sentence
    sentences = list(sentences)
    sentences = [re.sub(' +', ' ', sent.replace('"', '')) for sent in sentences]

    sentence_embeddings = model.encode(sentences)

    numpy.savetxt(SENTENCE_EMBEDDINGS, sentence_embeddings, delimiter=",")
    pd.DataFrame(sentences).to_csv(SENTENCES, index=False, header=False)

if __name__ == '__main__':
    generate_clinical_bert_representation()