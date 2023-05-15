import os
import ast
import time
from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import set_seed, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from datasets import Dataset
import mlflow
from mlflow.models.signature import infer_signature

from utils import FeatureEncoder, AverageMeter, TokenCLSForSpellCorrectionAPI


os.environ["TOKENIZERS_PARALLELISM"] = "false"
TYPO_WORDS = "typo_words"
LABELS = "labels"
INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
IGNORE_INDEX = -100


def evaluate(model, dataloader, device):
    losses = AverageMeter()
    model.eval()
    ground_true = []
    predict = []
    start_time = time.time()
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluating ... ", position=2)):
        batch_size = batch[LABELS].size(0)
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

        losses.update(loss.item(), batch_size)
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
    accuracy = sum([pre_seq == tar_seq for pre_seq, tar_seq in zip(predict, ground_true)]) / len(ground_true)
    duration = time.time() - start_time
    return losses.avg, accuracy, duration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_path", type=str, required=True, help="")
    parser.add_argument("--experiment_name", type=str, required=True, help="")
    parser.add_argument("--run_name", type=str, required=True, help="")
    parser.add_argument("--model_path", type=str, required=True, help="")
    parser.add_argument("--train_file", type=str, required=True, help="")
    parser.add_argument("--validation_file", type=str, required=True, help="")
    parser.add_argument("--test_file", default=None, type=str, help="")
    parser.add_argument("--vocab_file", type=str, required=True, help="")
    parser.add_argument("--log_file", default="train.log", type=str, help="")
    parser.add_argument("--pretrained_model_name_or_path", default="prajjwal1/bert-tiny", type=str, help="")
    parser.add_argument("--batch_size", default=16, type=int, help="")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="")
    parser.add_argument("--epochs", default=10, type=int, help="")
    parser.add_argument("--max_length", default=256, type=int, help="")
    parser.add_argument("--warmup_ratio", default=0.0, type=float, help="")
    parser.add_argument("--max_norm", default=1.0, type=float, help="")
    parser.add_argument("--accum_steps", default=1, type=int, help="")
    parser.add_argument("--seed", default=2330, type=int, help="")
    args = parser.parse_args()

    # ===== Set seed =====
    set_seed(args.seed)

    # ===== Set tracking URI =====
    EXPERIMENTS_PATH = Path(args.experiments_path)
    EXPERIMENTS_PATH.mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri(EXPERIMENTS_PATH)

    # ===== Set experiment =====
    mlflow.set_experiment(experiment_name=args.experiment_name)

    # ===== Load file =====
    df_train = pd.read_csv(args.train_file)
    df_val = pd.read_csv(args.validation_file)
    log_file = open(args.log_file, "w", encoding="utf-8")

    # ===== Preprocessing =====
    tqdm.pandas(desc="Convert data type (training file) ... ")
    df_train[TYPO_WORDS] = df_train[TYPO_WORDS].progress_apply(ast.literal_eval)
    df_train[LABELS] = df_train[LABELS].progress_apply(ast.literal_eval)

    tqdm.pandas(desc="Convert data type (validation file) ... ")
    df_val[TYPO_WORDS] = df_val[TYPO_WORDS].progress_apply(ast.literal_eval)
    df_val[LABELS] = df_val[LABELS].progress_apply(ast.literal_eval)

    train_dataset = Dataset.from_pandas(df_train)
    val_dataset = Dataset.from_pandas(df_val)

    feature_enccoder = FeatureEncoder(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        vocab_file=args.vocab_file,
        max_length=args.max_length,
    )
    feature_enccoder.tokenizer.save_pretrained(args.model_path)

    train_dataset = train_dataset.map(feature_enccoder.encode_with_label, batched=True)
    val_dataset = val_dataset.map(feature_enccoder.encode_with_label, batched=True)
    train_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    val_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # ===== Model =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=feature_enccoder.num_labels,
    ).to(device)

    # ===== Optimizer =====
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    ###################################
    ##########     Train     ##########
    ###################################

    # ===== Tracking =====
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(vars(args))
        best_val_accuracy = 0
        epochs = tqdm(range(args.epochs), desc="Epoch ... ", position=0)
        for epoch in epochs:
            model.train()
            train_losses = AverageMeter()
            start_time = time.time()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training ... ", position=1)):
                batch_size = batch[LABELS].size(0)
                for key in batch.keys():
                    batch[key] = batch[key].to(device)

                outputs = model(**batch)
                loss = outputs.loss
                train_losses.update(loss.item(), batch_size)

                if args.accum_steps > 1:
                    loss = loss / args.accum_steps
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

                if (step + 1) % args.accum_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if (step + 1) == 1 or (step + 1) % 2500 == 0 or (step + 1) == len(train_dataloader):
                    epochs.write(
                        f"Epoch: [{epoch + 1}][{step + 1}/{len(train_dataloader)}] "
                        f"Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) "
                        f"Grad: {grad_norm:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.8f}",
                        file=log_file,
                    )
                    log_file.flush()
                    os.fsync(log_file.fileno())

                    mlflow.log_metrics(
                        {
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=(len(train_dataloader) * epoch) + step,
                    )
            train_duration = time.time() - start_time
            epochs.write(f"Training Duration: {train_duration:.3f} sec", file=log_file)

            ####################################
            ##########   Validation   ##########
            ####################################

            val_loss, val_accuracy, val_duration = evaluate(model, val_dataloader, device)
            epochs.write(f"Validation Loss: {val_loss:.4f}", file=log_file)
            epochs.write(f"Validation Accuracy: {val_accuracy:.4f}", file=log_file)
            epochs.write(f"Validation Duration: {val_duration:.3f} sec", file=log_file)

            if val_accuracy > best_val_accuracy:
                model.save_pretrained(args.model_path)
                best_val_accuracy = val_accuracy

            mlflow.log_metrics(
                {
                    "train_loss": train_losses.avg,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "best_val_accuracy": best_val_accuracy,
                },
                step=epoch,
            )

        ####################################
        ##########      Test      ##########
        ####################################

        if args.test_file is not None:
            model = model.from_pretrained(args.model_path).to(device)
            df_test = pd.read_csv(args.test_file)
            tqdm.pandas(desc="Convert data type (test file) ... ")
            df_test[TYPO_WORDS] = df_test[TYPO_WORDS].progress_apply(ast.literal_eval)
            df_test[LABELS] = df_test[LABELS].progress_apply(ast.literal_eval)
            test_dataset = Dataset.from_pandas(df_test)
            test_dataset = test_dataset.map(feature_enccoder.encode_with_label, batched=True)
            test_dataset.set_format(type="torch", columns=[INPUT_IDS, ATTENTION_MASK, LABELS])
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

            test_loss, test_accuracy, test_duration = evaluate(model, test_dataloader, device)
            epochs.write(f"Test Loss: {test_loss:.4f}", file=log_file)
            epochs.write(f"Test Accuracy: {test_accuracy:.4f}", file=log_file)
            epochs.write(f"Test Duration: {test_duration:.3f} sec", file=log_file)

            mlflow.log_metrics(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                },
            )

        log_file.close()
        mlflow.log_artifact(args.log_file)

        # ===== Package model file to mlflow =====
        artifacts = {
            Path(file).stem: os.path.join(args.model_path, file)
            for file in os.listdir(args.model_path)
            if not os.path.basename(file).startswith(".")
        }
        artifacts["vocab_file"] = args.vocab_file

        sample = pd.DataFrame({"text": ["nodle in righ uppr lng .", "mass in lft loer lung ."]})
        results = pd.DataFrame({"results": ["nodule in right upper lung .", "mass in left lower lung ."]})
        signature = infer_signature(sample, results)

        mlflow.pyfunc.log_model(
            "model",
            python_model=TokenCLSForSpellCorrectionAPI(),
            code_path=["utils.py"],
            artifacts=artifacts,
            signature=signature,
        )


if __name__ == "__main__":
    main()
