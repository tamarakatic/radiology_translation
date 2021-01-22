import pandas as pd
from collections import defaultdict
from numpy import genfromtxt
from data.definition import SENTENCE_EMBEDDINGS

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy

def hierarchical_clustering():
    # Use huggingface/transformers pre-trained model Bio_ClinicalBERT for mapping tokens to embeddings
    sent_embeddings = genfromtxt(SENTENCE_EMBEDDINGS, delimiter=',')
    
    model = AgglomerativeClustering()
    model = model.fit(sent_embeddings)

    # distances = model.distances_
    num_clusters = model.n_clusters_

    print(f"Number of clusters: {num_clusters}")
    # print(f"Max distance between clusters: {distances.max()}")
    # print(f"Min distance between clusters: {distances.min()}")

if __name__ == '__main__':
    hierarchical_clustering()