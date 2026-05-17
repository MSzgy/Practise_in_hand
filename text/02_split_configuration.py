from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


MODEL_ID = "dphn/dolphin-2.9.4-llama3.1-8b"


config = AutoConfig.from_pretrained(MODEL_ID)
print("model_type:", config.model_type)
print("hidden_size:", getattr(config, "hidden_size", "unknown"))
print("num_hidden_layers:", getattr(config, "num_hidden_layers", "unknown"))
print("num_attention_heads:", getattr(config, "num_attention_heads", "unknown"))
print("num_key_value_heads:", getattr(config, "num_key_value_heads", "unknown"))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    padding_side="left",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    torch_dtype="auto",
    device_map="auto",
)

generation_config = GenerationConfig.from_pretrained(MODEL_ID)
generation_config.max_new_tokens = 120
generation_config.do_sample = True
generation_config.temperature = 0.7
generation_config.top_p = 0.9
generation_config.repetition_penalty = 1.05
generation_config.pad_token_id = tokenizer.pad_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

messages = [
    {"role": "system", "content": "你是一个面试辅导老师。"},
    {"role": "user", "content": "解释 prefill 和 decode 的区别。"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    generation_config=generation_config,
)

new_token_ids = outputs[0][inputs["input_ids"].shape[-1] :]
answer = tokenizer.decode(new_token_ids, skip_special_tokens=True)

print(answer)
