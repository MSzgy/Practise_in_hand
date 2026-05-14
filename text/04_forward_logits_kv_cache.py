import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "KV cache 是什么？"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

logits = outputs.logits[:, -1, :]
next_token_id = logits.argmax(dim=-1)
next_token = tokenizer.decode(next_token_id)

print("input token count:", inputs["input_ids"].shape[-1])
print("vocab size:", logits.shape[-1])
print("greedy next token id:", next_token_id.item())
print("greedy next token:", repr(next_token))

cache = outputs.past_key_values
print("KV cache type:", type(cache).__name__)

try:
    print("KV cache layers:", len(cache))
except TypeError:
    print("KV cache layers: unknown")

if hasattr(cache, "get_seq_length"):
    print("KV cache sequence length:", cache.get_seq_length())
