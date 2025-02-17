"""Finetune DeBERTa-v3 on the question-question pairs dataset."""
import os
from transformers import Trainer
from transformers import TrainingArguments
import transformers
import datasets

# Enviromment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["TRITON_CACHE_DIR"] = "/home/gasanoe/HOME_SCRATCH_FOLDER/.triton/autotune"

# Setting up constants
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128

# Dataset preparation
qqp = datasets.load_dataset('SetFit/qqp')
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    """A simple preprocess function for the qqp dataset."""
    result = tokenizer(
        examples['text1'], examples['text2'],
        padding='max_length', max_length=MAX_LENGTH, truncation=True
    )
    result['label'] = examples['label']
    return result

qqp_preprocessed = qqp.map(preprocess_function, batched=True)

# Model preparation
model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to('cuda')

# Setting up the training process arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    deepspeed="./ds_config.json",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=qqp_preprocessed["train"],
    eval_dataset=qqp_preprocessed["validation"],
    tokenizer=tokenizer
)

trainer.train()
