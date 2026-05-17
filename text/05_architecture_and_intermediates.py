import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "dphn/dolphin-2.9.4-llama3.1-8b"
OUTPUT_ATTENTIONS = False


def tensor_summary(value):
    if value is None:
        return "None"
    if hasattr(value, "shape"):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}"
    return type(value).__name__


def print_tensor_sequence(name, values):
    if values is None:
        print(f"{name}: None")
        return

    print(f"{name} count:", len(values))
    for index, value in enumerate(values):
        print(f"{name}[{index:02d}]:", tensor_summary(value))


def print_cache_summary(cache):
    if cache is None:
        print("KV cache: None")
        return

    print("KV cache type:", type(cache).__name__)

    try:
        print("KV cache layers:", len(cache))
    except TypeError:
        print("KV cache layers: unknown")

    if hasattr(cache, "get_seq_length"):
        print("KV cache sequence length:", cache.get_seq_length())

    if hasattr(cache, "key_cache") and getattr(cache, "key_cache"):
        print("layer0 key:", tensor_summary(cache.key_cache[0]))
        print("layer0 value:", tensor_summary(cache.value_cache[0]))
    elif isinstance(cache, (tuple, list)) and cache:
        first_layer = cache[0]
        if isinstance(first_layer, (tuple, list)) and len(first_layer) >= 2:
            print("layer0 key:", tensor_summary(first_layer[0]))
            print("layer0 value:", tensor_summary(first_layer[1]))


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "用一句话解释 hidden state 是什么？"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("=== Model Config ===")
print(model.config)

print("\n=== Model Architecture ===")
print(model)

with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=OUTPUT_ATTENTIONS,
        return_dict=True,
    )

last_token_logits = outputs.logits[:, -1, :]
next_token_id = last_token_logits.argmax(dim=-1)
next_token = tokenizer.decode(next_token_id)

print("\n=== Forward Outputs ===")
print("outputs type:", type(outputs).__name__)
print("outputs keys:", list(outputs.keys()))
print("input_ids:", tensor_summary(inputs["input_ids"]))
print("logits:", tensor_summary(outputs.logits))
print("last token logits:", tensor_summary(last_token_logits))
print("greedy next token id:", next_token_id.item())
print("greedy next token:", repr(next_token))

print("\n=== Hidden States ===")
print_tensor_sequence("hidden_states", outputs.hidden_states)

print("\n=== Attentions ===")
print_tensor_sequence("attentions", outputs.attentions)

print("\n=== KV Cache ===")
print_cache_summary(outputs.past_key_values)
