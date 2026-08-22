# PS -> Load the model in 4 bit and using cola gpu find out the vram usage
# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

# Configure 4-bit Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NormalFloat4 : optimal for normally distributed weights
    bnb_4bit_use_double_quant=True, # Quatizes the quantization constants (saves ~0.4 bits/param)
    bnb_4bit_compute_dtype=torch.bfloat16 # Computation happens in bfloat16 for numerical stability

)

# Load the tokenizer
tokenizer=AutoTokenizer.from_pretrained(model_id)

# Load model onto GPU (or CPU if CUDA is unavailable)
device_map = "auto" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = device_map,
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
)
# Calculate VRAM allocation
if torch.cuda.is_available():
    vram_used = torch.cuda.memory_allocated()/(1024**2)
    print(f"Model loaded successfully in 4-bit. VRAM allocated : {vram_used:.2f} MB") # Output was 2400 MB
else:
    print("CUDA not detected. Loaded on CPU.")
