import datasets
dataset = datasets.load_dataset(
    "race","high",
    split=["train", "validation", "test"],
)