from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

UNK_TOKEN = "[UNK]"


class TokenCLSForSpellCorrectionPipeline:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        vocab_file: str,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path).to(self.device)
        self.model.eval()
        self.unk_token_id = None
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.index_to_word = {i: line.strip() for i, line in enumerate(f.readlines())}

        for idx, word in self.index_to_word.items():
            if word == UNK_TOKEN:
                self.unk_token_id = idx
                break

        if self.unk_token_id is None:
            raise ValueError("Vocabulary file does not appear unknown token !")

    def __call__(self, sentence: str) -> str:
        words = sentence.split(" ")
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
        )
        torch.cuda.empty_cache()
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            pred_ids = torch.argmax(self.model(**inputs).logits, dim=-1).cpu().numpy()[0]

        outputs = []
        previous_word_idx = 0
        for pred_idx, word_idx in zip(pred_ids, inputs.word_ids()):
            if word_idx is not None and word_idx == previous_word_idx:
                if pred_idx == self.unk_token_id:
                    outputs.append(words[word_idx])
                else:
                    outputs.append(self.index_to_word[pred_idx])
                previous_word_idx += 1
        return " ".join(outputs)


if __name__ == "__main__":
    pipeline = TokenCLSForSpellCorrectionPipeline(
        "models/best_model",
        "program_data/vocab_file.txt",
    )
    print(pipeline("opciy and noodl at lung ."))
    print(pipeline("2. Mid lug emphysma ."))
    print(pipeline("# Atrophy of lebt kidney ."))
    print(pipeline("No pericaruial effusiwon ."))
