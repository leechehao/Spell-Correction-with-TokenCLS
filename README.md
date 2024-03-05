# Spell Correction with TokenCLS
透過 Token Classification 實現 Chest CT 影像文字報告的錯字校正。

## 資料格式
根據 4 種策略進行 Data Augmentation（資料增強）：
+ 插入字元
+ 移除字元
+ 替換字元
+ 相鄰字元互換
```
words,typo_words,labels,length
"['Mild', 'increased', 'amount', '.']","['bMild', 'increased', 'amout', '.']","[3165, 750, 324, 0]",4
"['Mild', 'increased', 'amount', '.']","['Mid', 'increased', 'amounzt', '.']","[3165, 750, 324, 0]",4
"['Mild', 'increased', 'amount', '.']","['Milwd', 'increased', 'aomunt', '.']","[3165, 750, 324, 0]",4
"['Mild', 'increased', 'amount', '.']","['Muild', 'increased', 'amnount', '.']","[3165, 750, 324, 0]",4
```

## 模型訓練
```
python 4_train.py --experiments_path PATH_TO_EXPERIMENTS \
                  --experiment_name EXPERIMENT_NAME \
                  --run_name RUN_NAME \
                  --model_path MODEL_SAVE_PATH \
                  --train_file TRAIN_DATASET_PATH \
                  --validation_file VALIDATION_DATASET_PATH \
                  --vocab_file VOCAB_FILE_PATH \
                  [--test_file TEST_DATASET_PATH] \
                  [--log_file LOG_FILE] \
                  [--pretrained_model_name_or_path PRETRAINED_MODEL] \
                  [--batch_size BATCH_SIZE] \
                  [--learning_rate LEARNING_RATE] \
                  [--epochs NUMBER_OF_EPOCHS] \
                  [--max_length MAX_TOKEN_LENGTH] \
                  [--warmup_ratio WARMUP_RATIO] \
                  [--max_norm MAX_GRADIENT_NORM] \
                  [--accum_steps GRADIENT_ACCUMULATION_STEPS] \
                  [--seed RANDOM_SEED]
```
### Required Arguments
+ **experiments_path** *(str)* ─ Path to the directory where experiment artifacts will be stored.
+ **experiment_name** *(str)* ─ Name of the MLflow experiment under which runs will be logged.
+ **run_name** *(str)* ─ Name of this run; used for logging purposes.
+ **model_path** *(str)* ─ Directory where the trained model and tokenizer will be saved.
+ **train_file** *(str)* ─ Path to the CSV file containing the training data.
+ **validation_file** *(str)* ─ Path to the CSV file containing the validation data.
+ **vocab_file** *(str)* ─ Path to the vocabulary file for the model output.


### Optional Arguments
+ **test_file** *(str, defaults to `None`)* ─ Path to the CSV file containing the test data.
+ **log_file** *(str, defaults to `train.log`)* ─ File where training logs will be written.
+ **pretrained_model_name_or_path** *(str, defaults to `prajjwal1/bert-tiny`)* ─ Pretrained model name or path to a pretrained model to be used for token classification.
+ **batch_size** *(int, defaults to `16`)* ─ Batch size for training and evaluation.
+ **learning_rate** *(float, defaults to `1e-4`)* ─ Learning rate for the optimizer.
+ **epochs** *(int, defaults to `10`)* ─ Number of epochs to train.
+ **max_length** *(int, defaults to `256`)* ─ Maximum sequence length for tokenization.
+ **warmup_ratio** *(float, defaults to `0.0`)* ─ Proportion of training to perform linear learning rate warmup.
+ **max_norm** *(float, defaults to `1.0`)* ─ Max norm for the gradients.
+ **accum_steps** *(int, defaults to `1`)* ─ Number of steps to accumulate gradients for.
+ **seed** *(int, defaults to `2330`)* ─ Random seed for initialization.

## 模型評估
```bash
python 5_test.py --test_file TEST_DATASET_PATH \
                 --vocab_file VOCAB_FILE_PATH \
                 --pretrained_model_name_or_path PRETRAINED_MODEL \
                 [--max_length MAX_TOKEN_LENGTH] \
                 [--eval_batch_size EVAL_BATCH_SIZE]
```
### Required Arguments
+ **test_file** *(str)* ─ Path to the CSV file containing the test data.
+ **vocab_file** *(str)* ─ Path to the vocabulary file for the model output.
+ **pretrained_model_name_or_path** *(str)* ─ The name or path of the pretrained model to be evaluated.

### Optional Arguments
+ **max_length** *(int, defaults to `128`)* ─ Maximum length of the input sequences.
+ **eval_batch_size** *(int, defaults to `100`)* ─ Batch size for evaluation.

### Evaluation Metrics
+ Test Loss: The average loss of the model on the test dataset.
+ Test Accuracy: Measures the percentage of correct predictions by the model on the test dataset.
+ Test Duration: The time taken to complete the evaluation.

## 模型推理 
```python
pipeline = TokenCLSForSpellCorrectionPipeline(
    "models/best_model",
    "program_data/vocab_file.txt",
)
print(pipeline("No pericaruial effusiwon ."))
```
輸出結果：
```
no pericardial effusion .
```

## 模型服務
透過指定的主機和端口啟動一個網絡伺服器，來部署一個用 MLflow 保存的模型。
```bash
#!/usr/bin/env sh

# Set environment variable for the tracking URL where the Model Registry resides
export MLFLOW_TRACKING_URI=http://localhost:9487

# Serve the production model from the model registry
mlflow models serve -m "models:/spell-correction/1" -h 0.0.0.0 -p 9488
```
API 使用方式請參考 `post.sh`。