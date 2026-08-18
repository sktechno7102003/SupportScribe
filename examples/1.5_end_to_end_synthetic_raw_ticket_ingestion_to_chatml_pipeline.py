# PS -> Given a raw pandas dataframe simulating uncleaned customer service logs from and ecommerce platfrom, construct a function
# that
# 1. validates raw classification targets using pydantic schema
# 2. cleans dirty ticket text
# 3. Formats each row into ChatML conversation (system, user, assistant)
# 4. Rejects any row that fails schema validation and outputs an execution summary

import pandas as pd
import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import re


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

# 2. Raw mock data (Containing one intentional invalid row for testing validation)
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
        "urgency_level": "INVALID_LEVEL", # This should fail validation! kyuki ye apne structure me hi hi nhi
        "primary_category": "BILLING",
        "sentiment": "NEGATIVE", 
        "extracted_entities": ["$49", "SUB-4412"],
        "recommended_action": "Forward to billing department."
    }
]

df_raw = pd.DataFrame(raw_data)

# 3. Processing Pipeline
def sanitize_ticket_text(text:str) -> str:
    """
    Cleans raw customer ticket text:
    1. Removes excess newlines and redundant spaces.
    2. Strips common email disclaimer footers.
    3. Normalizes smart quotes to standard ASCII quotes.
    """

    # Replace smart quotes to normal quotes
    text = text.replace("“", '"').replace("”", '"').replace("`", "'").replace("’", "'")

    #strip email signature boilerplate
    text = re.sub(r'(--\s*\n.*$)|(Sent from my iPhone.*$)|(Best regards.*$)','',text, flags = re.DOTALL | re.IGNORECASE)

    # Collapse multiple whitespaces/newlines to single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def build_instruction_dataset(df: pd.DataFrame, system_prompt:str):
    valid_training_rows = []
    rejected_rows = []

    for idx, row in df.iterrows(): # Har row pe ek ek karke iterate krega, yaha pe ek row ek mock data row hai
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

            # sanitize input text
            clean_text = sanitize_ticket_text(row["raw_text"])

            # Constructing chatml message block
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

print(f"Successfully processed: {len(valid_dataset)} rows")
print(f"Rejected rows : {len(errors)}")
print("\nSample Validated chatML Assistant Payload:")
print(valid_dataset[0])

# Successfully processed: 1 rows
# Rejected rows : 1

# Sample Validated chatML Assistant Payload:
# {'messages': [
#     {'role': 'system', 'content': """You are an enterprise support triage AI. You must analyze the 
#                                       incoming customer support ticket\n    and output a strictly valid JSON object 
#                                       matching the requested schema. Do not include markdown codeblocks or extra text.""""}, 
#     {'role': 'user', 'content': """Hey! My dashboard is completely blank on Chrome v120. Error code: ERR_NULL_POINTER. 
#                                       Fix ASAP!! sent from my Android"""}, 
#     {'role': 'assistant', 'content': """{"urgency_level" : "HIGH",
#                                         "primary_category" : "TECHNICAL_BUG",
#                                         "sentiment" : "FRUSTRATED",
#                                         "extracted_entities" : ["Chrome v120","ERR_NULL_POINTER"],
#                                         "recommended_action" : "Assign to frontend engineering on-call."}"""}]}

# Note that each row of our data is now converted to a chatml template and is compatible with SFTTrainer