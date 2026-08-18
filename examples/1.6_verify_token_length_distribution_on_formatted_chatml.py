# PS -> Before fine tuning you must ensure that your dataset does not exceed the model's context window or waste GPU
# memory with excessive padding. Write a function that takes the formatted chatMl messages, tokenizes them and prints
# 50th, 90th, and 99th percentile token lengths

import pandas as pd
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import re
import numpy as np
from transformers import AutoTokenizer


# 1. Defining the schema using pydantic
class SupportTicketSchema(BaseModel):
    urgency_level : Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Operational urgency based on busness impact"
    )
    primary_category : Literal["BILLING", "TECHNICAL_BUG", "ACCOUNT_ACCESS", "FEATURE_REQUEST"] = Field(
        description="Departmental routing category"
    )
    sentiment : Literal["POSITIVE", "NEUTRAL", "NEGATIVE","FRUSTRATED"] = Field(
        description="Detected user emotional state"
    )
    extracted_entities : List[str] = Field(
        default_factory=list, 
        description="Specific error codes, order IDs, or dates found in the text"
    )
    recommended_action : str = Field(
        description="one-Sentence action item for the support representative"
    )

SYSTEM_PROMPT = (
    """You are an enterprise support triage AI. You must analyze the incoming customer support ticket
    and output a strictly valid JSON object matching the requested schema. Do not include markdown codeblocks or extra text."""
)

raw_data = [
    {
        "ticket_id" : "TCK-101",
        "raw_text" : """Hey! My dashboard is completely blank on Chrome v120.
                    Error code: ERR_NULL_POINTER. Fix ASAP!! \n\nsent from my Android""",
        "urgency_level" : "HIGH", 
        "primary_category": "TECHNICAL_BUG",
        "sentiment" : "FRUSTRATED", 
        "extracted_entities": ["Chrome v120", "ERR_NULL_POINTER"],
        "recommended_action" : "Assign to frontend engineering on-call."
    },
    {
        "ticket_id": "TCK-102",
        "raw_text": "I was overcharged $49 on subscription SUB-4412. Refund please.",
        "urgency_level": "HIGH",
        "primary_category": "BILLING",
        "sentiment": "NEGATIVE", 
        "extracted_entities": ["$49", "SUB-4412"],
        "recommended_action": "Forward to billing department."
    }
]

df_raw = pd.DataFrame(raw_data)

def sanitize_ticket_text(text:str) -> str:
    """
    Cleans raw customer ticket text:
    1. Removes excess newlines and redundant spaces.
    2. Strips common email disclaimer footers.
    3. Normalizes smart quotes to standard ASCII quotes.
    """

    text = text.replace("“", '"').replace("”", '"').replace("`", "'").replace("’", "'")
    text = re.sub(r'(--\s*\n.*$)|(Sent from my iPhone.*$)|(Best regards.*$)','',text, flags = re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_instruction_dataset(df: pd.DataFrame, system_prompt:str):
    valid_training_rows = []
    rejected_rows = []

    for idx, row in df.iterrows(): 
        try:
            # validate target payload
            target_data = {
                "urgency_level" : row["urgency_level"],
                "primary_category" : row["primary_category"],
                "sentiment" : row["sentiment"],
                "extracted_entities" : row["extracted_entities"],
                "recommended_action" : row["recommended_action"]
            }
            validated_payload = SupportTicketSchema(**target_data)
            clean_text = sanitize_ticket_text(row["raw_text"])
            message_block = {
                "messages" : [
                    {"role":"system", "content":SYSTEM_PROMPT},
                    {"role":"user", "content": clean_text},
                    {"role" : "assistant", "content" : validated_payload.model_dump_json()}
                ]
            }
            valid_training_rows.append(message_block)
        except ValidationError as e:
            rejected_rows.append({"index" : idx, "ticket_id" : row["ticket_id"], "error" : str(e)})
    return valid_training_rows, rejected_rows

# Execute
valid_dataset, errors = build_instruction_dataset(df_raw, SYSTEM_PROMPT)


model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
def analyze_token_lengths(formatted_dataset: list, model_id: str = model_id):
    token_lengths = []
    for item in formatted_dataset:
        # tokenize = true direct returns token ids
        input_ids = tokenizer.apply_chat_template(item["messages"], tokenize = True, return_dict = False)
        # return dict = false krne se attention mask is not given only input tokens ke ids ka list aayega
        # while passing through SFTTrainer we will not do return_dict - false, bcs waha pe we need the attention mask as well
        token_lengths.append(len(input_ids))
    lengths_array = np.array(token_lengths)

    metrics = {
        "Total Samples": len(lengths_array),
        "Min Length" : int(np.min(lengths_array)),
        "Mean Length" : float(np.mean(lengths_array)),
        "50th Percentile" : float(np.percentile(lengths_array, 50)),
        "95th Percentile" : float(np.percentile(lengths_array, 95)),
        "Max Length" : int(np.max(lengths_array))
    }
    return metrics

# Test with our valid dataset from previous example
metrics = analyze_token_lengths(valid_dataset)
for k,v in metrics.items():
    print(f"{k}:{v}")


# Output
# Total Samples:2
# Min Length:119
# Mean Length:126.5
# 50th Percentile:126.5
# 95th Percentile:133.25
# Max Length:134