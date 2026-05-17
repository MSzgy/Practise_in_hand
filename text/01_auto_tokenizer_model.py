from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "dphn/dolphin-2.9.4-llama3.1-8b"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "你是一个讲解大模型部署的老师。"},
    {"role": "user", "content": "用三句话解释什么是 chat template。"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
)

new_token_ids = outputs[0][inputs["input_ids"].shape[-1] :]
answer = tokenizer.decode(new_token_ids, skip_special_tokens=True)

print(answer)
