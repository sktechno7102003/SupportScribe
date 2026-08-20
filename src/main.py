from generator import generate_mock_tickets
from validator import validate_target
from sanitizer import clean_text
from formatter import get_token_lengths, format_to_chatml
import pandas as pd
from datasets import Dataset
import numpy as np
import os
import json


SYSTEM_PROMPT = (
    """You are an enterprise support triage AI. You must analyze the incoming customer support ticket
    and output a strictly valid JSON object matching the requested schema. Do not include markdown codeblocks or extra text."""
    )

def run_pipeline():
    # Generating the  250 raw customer tickets
    raw_tickets = generate_mock_tickets(250)
    lengths_array = []
    transformed_records = []
    # validating the pydantic schema and cleaning the raw_text to convert in chatMl format
    for query in raw_tickets:
        if validate_target(query)[1]:
            # 1. validating schema using pydantic and returning serialized json string
            validated_dict = validate_target(query)[0]
            # 2. Cleaning raw_user_text
            cleaned_text = clean_text(query["raw_text"])
            # 3. Formatting the text into chatml format
            formatted_text = format_to_chatml(SYSTEM_PROMPT, cleaned_text, validated_dict)
            # 4. tokenizing the input using apply_chat_template to get total number of tokens
            length = get_token_lengths(formatted_text)
            lengths_array.append(length)
            # 5. applying 512 token limit threshold
            if length <= 1024:
                transformed_records.append(formatted_text)
    # saving Token length report
    metrics = {
        "Total Samples": len(lengths_array),
        "Min Length" : int(np.min(lengths_array)),
        "Mean Length" : float(np.mean(lengths_array)),
        "50th Percentile" : float(np.percentile(lengths_array, 50)),
        "95th Percentile" : float(np.percentile(lengths_array, 95)),
        "Max Length" : int(np.max(lengths_array))
    }
    with open("reports/token_distribution.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # converting to huggingface dataset object & train-test split
    df = pd.DataFrame(transformed_records)
    hf_dataset = Dataset.from_pandas(df)
    # saving train and test records in jsonl format in disk
    dataset_split = hf_dataset.train_test_split(test_size=0.2, seed = 42)
    dataset_split["train"].to_json("data/train.jsonl", orient="records", lines=True)
    dataset_split["test"].to_json("data/val.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    run_pipeline()
