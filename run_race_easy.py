MAX_SEQ_LENGTH=1024
import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import datasets
from transformers import RobertaTokenizer
from transformers import  RobertaForMultipleChoice
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import transformers
import numpy as np

logger = logging.getLogger(__name__)
tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
raw_dataset = datasets.load_dataset(
    "race", "high",
)
#加载race

def convert_ans_to_label(ans):
    dicts = {"A":0, "B":1, "C":2, "D":3}
    return [dicts[e] for e in ans]
dataset = raw_dataset.map(lambda x: {"label": convert_ans_to_label(x["answer"])}, batched=True)
#添加label项(use map func)

train_dataset=dataset["train"],
eval_dataset=dataset["validation"]

def preprocess_function(examples):
    contexts = [[context] * 4 for context in examples["article"]]
    question_headers = examples["question"]
    options = examples["options"]
    answersents = [
        [f"{header} {options[idx][i]}" for i in range(4)] for idx, header in enumerate(question_headers)
    ]

    contexts = sum(contexts, [])
    answersents = sum(answersents, [])

    tokenized_examples = tokenizer(contexts, answersents, truncation=True)
    return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
#预处理


transformers.logging.set_verbosity_error()
#减少不必要的警告

tokenized_race = dataset.map(preprocess_function, batched=True)
#tokenization



#下面是data collator
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
#计算准确度
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

#可以train了
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    fp16=True,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_race["train"],
    eval_dataset=tokenized_race["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

train_result=trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload
metrics = train_result.metrics

metrics["train_samples"] = len(train_dataset)
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
#Evaluate
logger.info("*** Evaluate ***")

metrics = trainer.evaluate()

metrics["eval_samples"] = len(eval_dataset)

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
