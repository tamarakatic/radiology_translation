import pandas as pd
import numpy as np
import re
from collections import defaultdict
import collections
from data.definition import LIVER_FINDINGS
from data.definition import TEMPLE_RADIO_1_2_SENTENCE_EMBEDDINGS, TEMPLE_RADIO_1_2_SENTENCES_WITH_TRANSLATION, DUPL_SENTENCES

from sentence_transformers import SentenceTransformer

def generate_clinical_bert_representation():
    # Use huggingface/transformers pre-trained model Bio_ClinicalBERT for mapping tokens to embeddings
    model = SentenceTransformer("emilyalsentzer/Bio_ClinicalBERT")

    first_sentences = pd.read_csv(LIVER_FINDINGS, header=0, usecols =["Sentence 1", "Sentence 1 Translation"])
    second_sentences = pd.read_csv(LIVER_FINDINGS, header=0, usecols =["Sentence 2", "Sentence 2 Translation"])
    first_sentences.columns = ["Sentence", "Translation"]
    second_sentences.columns = ["Sentence", "Translation"]

    df = pd.concat([first_sentences, second_sentences]).reset_index(drop=True)
    df = df[df['Sentence'].notna()]
    df['Translation'].replace(np.nan, 'empty', regex=True, inplace=True)   
    df = df.drop_duplicates(subset=['Sentence'], keep='first')

    df['Sentence'] = [sent if sent.endswith('.') else sent + '.' for sent in df['Sentence'].values]
    df.Translation = [sent.strip() for sent in df.Translation.values]
    df['Sentence'] = [re.sub('(\d{1,2})/(\d{1,2})/(\d{4})', '', sent) for sent in df['Sentence'].values]
    df['Sentence'] = [re.sub('[0-9]+.[0-9]+', '', sent) for sent in df['Sentence'].values]
    df['Sentence'] = [re.sub('[0-9]+', '', sent) for sent in df['Sentence'].values]
    df['Sentence'] = [sent.replace('  ', ' ') for sent in df['Sentence'].values]
    df['Sentence'] = [sent.strip() for sent in df['Sentence'].values]

    df.to_csv(TEMPLE_RADIO_1_2_SENTENCES_WITH_TRANSLATION, index=False)

    df['Sentence'] = [sent.lower() for sent in df['Sentence'].values]
    sentence_embeddings = model.encode(df.Sentence.to_list())

    np.savetxt(TEMPLE_RADIO_1_2_SENTENCE_EMBEDDINGS, sentence_embeddings, delimiter=",")
    

if __name__ == '__main__':
    generate_clinical_bert_representation()