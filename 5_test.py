import ast
import argparse
import time

import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
import datasets

from utils import FeatureEncoder, AverageMeter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="The path to a file containing the test data.")
    parser.add_argument("--vocab_file", type=str, required=True, help="")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="The name or path of a pretrained model that the script should use.")
    parser.add_argument("--max_length", default=128, type=int, help="The maximum length of the input sequences.")
    parser.add_argument("--eval_batch_size", default=100, type=int, help="The batch size to use during evaluation.")
    args = parser.parse_args()

    TYPO_WORDS = "typo_words"
    LABELS = "labels"
    INPUT_IDS = "input_ids"
    ATTENTION_MASK = "attention_mask"
    IGNORE_INDEX = -100

    # ===== 載入模型 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model_name_or_path).to(device)
    model.eval()

    # ===== 讀取資料 =====
    df_test = pd.read_csv(args.test_file)
    tqdm.pandas(desc="Convert data type (test file) ... ")
    df_test[TYPO_WORDS] = df_test[TYPO_WORDS].progress_apply(ast.literal_eval)
    df_test[LABELS] = df_test[LABELS].progress_apply(ast.literal_eval)

    feature_enccoder = FeatureEncoder(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        vocab_file=args.vocab_file,
        max_length=args.max_length,
    )
    test_dataset = datasets.Dataset.from_pandas(df_test)
    test_dataset = test_dataset.map(feature_enccoder.encode_with_label, batched=True)
    test_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    test_losses = AverageMeter()
    ground_true = []
    predict = []
    start_time = time.time()
    for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating ... ", position=2)):
        batch_size = batch[LABELS].size(0)

        with torch.no_grad():
            outputs = model(
                input_ids=batch[INPUT_IDS].to(device),
                attention_mask=batch[ATTENTION_MASK].to(device),
                labels=batch[LABELS].to(device),
            )
            loss = outputs.loss

        test_losses.update(loss.item(), batch_size)
        ground_true.extend(
            [
                [idx for idx in seq_idx if idx != IGNORE_INDEX]
                for seq_idx in batch[LABELS].numpy().tolist()
            ]
        )
        predict.extend(
            [
                [pre_idx for (pre_idx, tar_idx) in zip(pre_seq, tar_seq) if tar_idx != IGNORE_INDEX]
                for (pre_seq, tar_seq) in zip(torch.argmax(outputs.logits, dim=2).cpu().numpy().tolist(), batch[LABELS])
            ]
        )
    test_accuracy = sum([pre_seq == tar_seq for pre_seq, tar_seq in zip(predict, ground_true)]) / len(ground_true)
    test_duration = time.time() - start_time
    print(f"Test Loss: {test_losses.avg:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Duration: {test_duration:.3f} sec")


if __name__ == "__main__":
    main()
