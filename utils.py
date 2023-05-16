import os

import torch
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
import datasets
from mlflow.pyfunc import PythonModel, PythonModelContext


TYPO_WORDS = "typo_words"
MAX_LENGTH = "max_length"
LABELS = "labels"
IGNORE_INDEX = -100
UNK_TOKEN = "[UNK]"


class FeatureEncoder(object):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        vocab_file: str,
        max_length: int,
    ) -> None:
        self.tokenizer: transformers.PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
        )
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.num_labels = len(f.readlines())
        self.max_length = max_length

    def encode_with_label(self, examples: datasets.formatting.formatting.LazyBatch) -> transformers.BatchEncoding:
        tokenized_inputs = self.tokenizer(
            examples[TYPO_WORDS],
            padding=MAX_LENGTH,
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[LABELS]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is not None and word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(IGNORE_INDEX)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs[LABELS] = labels
        return tokenized_inputs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TokenCLSForSpellCorrectionAPI(PythonModel):
    def load_context(self, context: PythonModelContext):
        model_path = os.path.dirname(context.artifacts["config"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.model.eval()
        self.unk_token_id = None
        with open(context.artifacts["vocab_file"], "r", encoding="utf-8") as f:
            self.index_to_word = {i: line.strip() for i, line in enumerate(f.readlines())}

        for idx, word in self.index_to_word.items():
            if word == UNK_TOKEN:
                self.unk_token_id = idx
                break

        if self.unk_token_id is None:
            raise ValueError("Vocabulary file does not appear unknown token !")

    def predict(self, context: PythonModelContext, df: pd.DataFrame) -> pd.DataFrame:
        data = df.text.apply(lambda x: x.split(" ")).values.tolist()
        inputs = self.tokenizer(
            data,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        if self.model.device.index != None:
            torch.cuda.empty_cache()
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.model.device.index)

        with torch.no_grad():
            prediction = torch.argmax(self.model(**inputs).logits, dim=-1).cpu().numpy()

        results = []
        for i, pred_ids in enumerate(prediction):
            outputs = []
            previous_word_idx = 0

            for pred_idx, word_idx in zip(pred_ids, inputs.word_ids(batch_index=i)):
                if word_idx is not None and word_idx == previous_word_idx:
                    if pred_idx == self.unk_token_id:
                        outputs.append(data[i][word_idx])
                    else:
                        outputs.append(self.index_to_word[pred_idx])
                    previous_word_idx += 1
            results.append(" ".join(outputs))

        return pd.DataFrame({"results": results})
