# PS -> Hugging face SFT trainer expects a HuggingFace Dataset object. convert you dataset into dataframe first and then 
# after doing the train validation split save in the disk in jsonl format

import pandas as pd
from datasets import Dataset
import json

SYSTEM_PROMPT = (
    """You are an enterprise support triage AI. You must analyze the incoming customer support ticket
    and output a strictly valid JSON object matching the requested schema. Do not include markdown codeblocks or extra text."""
)

# Sample list of few structured records
records = [
    {
        "messages": [ 
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Payment failed for invoice #102."},
            {"role": "assistant", "content": json. dumps({"urgency_level": "HIGH", 
                                                          "primary_category": "BILLING",
                                                          "sentiment": "NEUTRAL", 
                                                          "extracted_entities": ["#102"], 
                                                          "recommended_action": "Check Stripe gateway logs. "})}
        ]
    },
    {
        "messages": [ 
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Can we get a dark mode feature"},
                {"role": "assistant", "content": json. dumps({"urgency_level": "LOW", 
                                                              "primary_category": "FEATURE_REQUEST",
                                                              "sentiment": "POSITIVE", 
                                                              "extracted_entities": ["dark_mode"], 
                                                              "recommended_action": "Log ticket to product backlog in Jira."})}
        ]
    }
]

# Converting list to pandas dataFrame, then to HuggingFace Dataset
df = pd.DataFrame(records)
hf_dataset = Dataset.from_pandas(df)

# Performing 80/20 Validation split
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed = 42)

# Save to disk as JSONL
#dataset_split["train"].to_json("train_dataset.json", orient="records", lines=True)
#dataset_split["test"].to_json("val_dataset.json", orient="records", lines=True)

print("Dataset Split Summary : ")
print(dataset_split)

# Output

# Dataset Split Summary : 
# DatasetDict({
#     train: Dataset({
#         features: ['messages'],
#         num_rows: 1
#     })
#     test: Dataset({
#         features: ['messages'],
#         num_rows: 1
#     })
# })

# NOte that yaha pe apna hf dataset jo bana uske andar do dataset hai ek train and dusra test