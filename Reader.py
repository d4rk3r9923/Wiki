import CFG
import numpy as np
import collections
from transformers import AutoTokenizer
from tqdm import tqdm

class Reader:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples['question']]
        inputs = self.tokenizer(questions,
                               examples['context'],
                               max_length=CFG.MAX_LENGTH,
                               truncation='only_second',
                               stride=CFG.STRIDE,
                               return_overflowing_tokens=True,
                               return_offsets_mapping=True,
                               padding='max_length')
        offset_mapping = inputs.pop('offset_mapping')
        sample_map = inputs.pop('overflow_to_sample_mapping')
        answers = examples['answers']
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            answer = answers[sample_idx]
            if len(answer['text']) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answer['answer_start'][0]
                end_char = answer['answer_start'][0] + len(answer['text'][0])
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                        start_positions.append(idx - 1)
                        idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions
        return inputs
    
    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples['question']]
        inputs = self.tokenizer(questions,
                               examples['context'],
                               max_length=CFG.MAX_LENGTH,
                               truncation='only_second',
                               stride=CFG.STRIDE,
                               return_overflowing_tokens=True,
                               return_offsets_mapping=True,
                               padding='max_length')
        sample_map = inputs.pop('overflow_to_sample_mapping')
        example_ids = []
        for i in range(len(inputs['input_ids'])):
            sample_idx = sample_map[i]
            example_ids.append(examples['id'][sample_idx])
            sequence_ids = inputs.sequence_ids(i)
            offset = inputs['offset_mapping'][i]
            inputs['offset_mapping'][i] = [x if sequence_ids[k] == 1 else None for k, x in enumerate(offset)]
        inputs['example_id'] = example_ids
        return inputs
    
    def compute_metrics(self, metric, start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature['example_id']].append(idx)
        predicted_answers = []
        for example in tqdm(examples):
            example_id = example['id']
            context = example ['context']
            answers = []
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]['offset_mapping']
                start_indexes = np.argsort(start_logit)[-1:-CFG.N_BEST - 1:-1].tolist()
                end_indexes = np.argsort(end_logit)[-1:-CFG.N_BEST - 1:-1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        if end_index - start_index + 1 > CFG.MAX_ANS_LENGTH:
                            continue
                        answer = {
                            'text': context[offsets[start_index][0]:offsets[end_index][1]],
                            'logit_score': CFG.start_logit[start_index] + end_logit[end_index]
                            }
                        answers.append(answer)
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x['logit_score'])
                answer_dict = {
                    'id': example_id,
                    'prediction_text': best_answer['text'],
                    'no_answer_probability': 1 - best_answer['logit_score']
                    }
            else:
                answer_dict = {
                    'id': example_id,
                    'prediction_text': '',
                    'no_answer_probability': 1.0
                    }
            predicted_answers.append(answer_dict)
        theoretical_answers = [
            {'id': ex['id'],
             'answers': ex['answers']
             } for ex in examples
             ]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)
