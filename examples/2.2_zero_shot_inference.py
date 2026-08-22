# PS -> Using the 4-bit model, perform zero shot inference on a single customer ticket
# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer=AutoTokenizer.from_pretrained(model_id)
device_map = "auto" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = device_map,
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

def run_zero_shot_inference(model, tokenizer, system_prompt:str, user_ticket:str) -> str:
    """Executes greedy inference on a single customer ticket without ground truth labels."""
    messages = [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":user_ticket}
    ] # Ye input format ready kr rhe hai

    # Format using tokenizer's ChatML template (add_generation_prompt=True appends '<|im_start|>assistant\n')
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True # ye <|im_start|>assistant ye append kiya so that model iske aage completion kare
    ) # ye input ko model specific format me convert kr rha hai
    inputs = tokenizer(prompt_text, return_tensors = "pt").to(model.device) # PyTorch tensor me convert krke gpu pe bhej diya
    with torch.no_grad(): # Disabling torch gradient tracking
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256, # output limit set kr rha, LLM kayi baar infintely \n likh dete hai usko avoid krega ye
            do_sample=False, # Greedy search for deterministic benchmark
            pad_token_id=tokenizer.eos_token_id
        ) #output me input+generated output aata hai so input tokens ko hatake decode krna padega
    # Slice off the input prompt tokens to leave only generated response
    input_length = inputs["input_ids"].shape[1] # input tokens ka lenght nikaal rhe to remove
    generated_tokens = output_ids[0][input_length:] # input lenght ke aage jo hai wo generated content hai

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() # decoding the generated content
    return response_text

# Practical test Run
test_system_prompt = (
    """
You are an enterprise support triage AI. You must analyze the customer support ticket and output a strictly valid JSON
object mathcing this schema :
{
"urgency_level":"LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
"primary_category" : "BILLING"| "TECHNICAL_BUG"| "ACCOUNT_ACCESS"| "FEATURE_REQUEST",
"sentiment" : "POSITIVE" | "NEUTRAL" | "NEGATIVE" | "FRUSTRATED",
"extracted_entities" : list of strings,
"recommended_action" : string
}
Output only the JSON object. Do not include markdown codeblocks or commentary.
"""
)

sample_ticket = "Payment failed twice for invoice INV-9901! I need access right now or i will cancel!"
output = run_zero_shot_inference(model, tokenizer, test_system_prompt, sample_ticket)
print("Raw Model Output :")
print(output)

# output-> json file format strictly follow nhi kr rha prompting ke baad v, unse mardown me wrap kr diya jabki mana kiya tha
# Reason is ki SLM kaafi training side biased rahte hai

# Raw Model Output :


# ```json
# {
#   "urgency_level": "HIGH",
#   "primary_category": "TECHNICAL_BUG",
#   "sentiment": "FRUSTRATED",
#   "extracted_entities": [
#     "INV-9901"
#   ],
#   "recommended_action": "Provide immediate access to resolve the payment issue."     # Notice the problem here
# }
# ```
# %%
