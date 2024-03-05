import CFG
from Retriever import get_embeddings_dataset
from transformers import pipeline, AutoModelForQuestionAnswering
from utils import get_embeddings

if __name__ == "__main__":
    model = AutoModelForQuestionAnswering.from_pretrained(CFG.FINETUNED_MODEL_NAME)
    tokenizer = CFG.FINETUNED_MODEL_NAME
    pipe = pipeline(CFG.PIPELINE_NAME, model=model, tokenizer=tokenizer)
    answer = pipe(question=CFG.INPUT_QUESTION, context=CFG.INPUT_CONTEXT)
    print(answer)
