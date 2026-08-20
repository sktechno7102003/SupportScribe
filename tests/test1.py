import os
import json
from datasets import load_dataset

def test_files_exist():
    assert os.path.exists("data/train.jsonl"), "train.jsonl is missing!"
    assert os.path.exists("data/val.jsonl"), "val.jsonl is missing!"

def test_huggingface_dataset_loadable():
    # If this fails, your JSONL formatting is broken
    dataset = load_dataset("json", data_files={"train": "data/train.jsonl", "val": "data/val.jsonl"})
    assert len(dataset["train"]) > 0
    assert len(dataset["val"]) > 0

def test_chatml_formatting():
    with open("data/train.jsonl", "r") as f:
        first_row = json.loads(f.readline())
        text = first_row.get("0", "") # your pipeline maps the formatted string to a '0' key
        assert "<|im_start|>system" in text, "Missing ChatML system token"
        assert "<|im_end|>" in text, "Missing ChatML end token"
        assert "<|im_start|>user" in text, "Missing ChatML user token"

# Running test1 to verify whether dps1 has been solved successfully
test_files_exist()
print()
test_huggingface_dataset_loadable()
print()
test_chatml_formatting()