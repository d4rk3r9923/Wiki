import CFG
from Retriever import get_embeddings_dataset
from transformers import pipeline, AutoModelForQuestionAnswering
from utils import get_embeddings

if __name__ == "__main__":
    model = AutoModelForQuestionAnswering.from_pretrained(CFG.FINETUNED_MODEL_NAME)
    tokenizer = CFG.FINETUNED_MODEL_NAME
    pipe = pipeline(CFG.PIPELINE_NAME, model=model, tokenizer=tokenizer)
    index = get_embeddings_dataset()
    scores, samples = index.get_nearest_examples(
        CFG.EMBEDDING_COLUMN,
        get_embeddings([CFG.INPUT_QUESTION]).cpu().detach().numpy(),
        k=CFG.TOP_K
    )
    print(f'Question: {CFG.INPUT_QUESTION}')
    for idx, score in enumerate(scores):
        question = samples['question'][idx]
        context = samples['context'][idx]
        answer = pipe(question=question, context=context)
        print(f'Top {idx + 1}:')
        print(f'Score: {score}')
        print(f'Context: {context}')
        print(f'Answer: {answer}')
        # for key in answer:
        #     print(f"- {key} : {answer[key]}")
        print()
