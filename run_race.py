MAX_SEQ_LENGTH = 1024
import datasets
from transformers import RobertaTokenizer
from transformers import  RobertaForMultipleChoice

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/aristo-roberta")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/aristo-roberta")
dataset = datasets.load_dataset(
    "race","high",
    split=["train", "validation", "test"],
)
training_examples = dataset[0]
evaluation_examples = dataset[1]
test_examples = dataset[2]

example=training_examples[0] 
example_id = example["example_id"]
question = example["question"]
label_example = example["answer"]
options = example["options"]
context = example["article"]
if label_example in ["A", "B", "C", "D"]:
    label_map = {label: i for i, label in enumerate(
                    ["A", "B", "C", "D"])}
elif label_example in ["1", "2", "3", "4"]:
    label_map = {label: i for i, label in enumerate(
                    ["1", "2", "3", "4"])}
else:
    print(f"{label_example} not found")
choices_inputs = []
for ending_idx, (_, ending) in enumerate(zip(context, options)):
    if question.find("_") != -1:
        # fill in the banks questions
        question_option = question.replace("_", ending)
    else:
        question_option = question + " " + ending
    
    inputs = tokenizer(
        context,
        question_option,
        add_special_tokens=True,
        
        padding=True,
        truncation=True,
        return_overflowing_tokens=False,
        return_tensors="pt"
    )
    choices_inputs.append(inputs)
    
label = label_map[label_example]
input_ids = [x["input_ids"] for x in choices_inputs]
attention_mask = (
    [x["attention_mask"] for x in choices_inputs]
     # as the senteces follow the same structure, just one of them is
     # necessary to check
   
)
example_encoded = {
    "input_ids": input_ids[0],
    "attention_mask": attention_mask[0],
}
output = model(**example_encoded)
