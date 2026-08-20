import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import numpy as np
from transformers import AutoTokenizer

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def format_to_chatml(SYSTEM_PROMPT, clean_user_text, target_json_string):
    message_block = {
        "messages" : [
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content": clean_user_text},
            {"role" : "assistant", "content" : target_json_string}
        ]
    }
    return(message_block)


def get_token_lengths(formatted_text: str, model_id: str = model_id):
    # tokenize = true direct returns token ids
    input_ids = tokenizer.apply_chat_template(formatted_text["messages"], tokenize = False)
    return len(input_ids)
