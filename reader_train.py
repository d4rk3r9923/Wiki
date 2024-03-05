import CFG
import torch
import evaluate
from Reader import Reader
from datasets import load_dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer

if __name__ == "__main__":
    reader = Reader()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    raw_datasets = load_dataset(CFG.DATASET_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(CFG.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)
    train_dataset = raw_datasets['train'].map(
        reader.preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets['train'].column_names
        )
    validation_dataset = raw_datasets['validation'].map(
        reader.preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets['validation'].column_names
        )
    args = TrainingArguments(
        output_dir='distilibert-finetuned-squadv2',
        evaluation_strategy='no',
        save_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        fp16=True,
        push_to_hub=True
        )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer
        )
    trainer.train()
    trainer.push_to_hub(commit_message='Training completed.')
    metric = evaluate.load(CFG.DATASET_NAME)
    predictions, _, _ = trainer.predict(validation_dataset)
    start_logits, end_logits = predictions
    results = reader.compute_metrics(metric, start_logits, end_logits, validation_dataset, raw_datasets['validation'])
    print(results)
