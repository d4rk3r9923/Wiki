import CFG
import os
from datasets import load_dataset
from utils import get_embeddings

def get_embeddings_dataset():
    raw_datasets = load_dataset(CFG.DATASET_NAME, split='train+validation')
    raw_datasets = raw_datasets.filter(lambda x: len(x['answers']['text']) > 0)
    embeddings_dataset = raw_datasets.map(lambda x: {CFG.EMBEDDING_COLUMN: get_embeddings(x['question']).detach().cpu().numpy()[0]})
    embeddings_dataset.add_faiss_index(column=CFG.EMBEDDING_COLUMN)
    return embeddings_dataset
