# Training configs
MODEL_NAME = 'distilbert-base-uncased'
DATASET_NAME = 'squad_v2'
MAX_LENGTH = 384
STRIDE = 128

# Validating configs
N_BEST = 20
MAX_ANS_LENGTH = 30

# Testing configs
PIPELINE_NAME = 'question-answering'
FINETUNED_MODEL_NAME = 'd4rk3r/distilibert-finetuned-squadv2'
INPUT_QUESTION = 'When did Beyonce start becoming popular?'
INPUT_CONTEXT = 'The quick brown fox jumps over the lazy dog.'

# Retriever configs
EMBEDDING_COLUMN = 'question_embedding'
TOP_K = 5
