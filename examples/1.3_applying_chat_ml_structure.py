# PS -> Since models like qwen 2.5 expect conversation to follow a standardized dictionary format, convert you text in that.

from transformers import AutoTokenizer # isse apne text to chat template me convert karenge and control tokens inject karenge
import json
# Load tokenizer for Qwen 2.5 1.5B instruct (Runs locally on CPU memory, no GPU needed for tokenization)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

SYSTEM_PROMPT = (
    """You are an enterprise support triage AI. You must analyze the incoming customer support ticket
    and output a strictly valid JSON object matching the requested schema. Do not include markdown codeblocks or extra text."""
)

raw_user_ticket = "I am locked out of my corporate account since 9 AM. Error: AUTH_TIMEOUT."

target_json = json.dumps({
    "urgency_level" : "HIGH",
    "primary_category" : "ACCOUNT_ACCESS",
    "sentiment" : "NEGATIVE",
    "extracted_entities" : ["AUTH_TIMEOUT"],
    "recommended_action" : "Trigger automated password/MFA reset link to verified corporate email."
})

messages = [
    {"role":"system", "content":SYSTEM_PROMPT},
    {"role":"user", "content":raw_user_ticket},
    {"role" : "assistant", "content" : target_json}
]

# Render the formatted string using the model's native ChatML template

formatted_chatml = tokenizer.apply_chat_template(messages, tokenize = False)
print("Formatted ChatML string with Control tokens:")
print(formatted_chatml)

# output

# Formatted ChatML string with Control tokens:
# <|im_start|>system
# You are an enterprise support triage AI. You must analyze the incoming customer support ticket
#     and output a strictly valid JSON object matching the requested schema. Do not include markdown codeblocks or extra text.<|im_end|>
# <|im_start|>user
# I am locked out of my corporate account since 9 AM. Error: AUTH_TIMEOUT.<|im_end|>
# <|im_start|>assistant
# {"urgency_level": "HIGH",
#  "primary_category": "ACCOUNT_ACCESS", 
#  "sentiment": "NEGATIVE", 
#  "extracted_entities": ["AUTH_TIMEOUT"], 
#  "recommended_action": "Trigger automated password/MFA reset link to verified corporate email."}<|im_end|>