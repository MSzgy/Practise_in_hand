from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class StopOnTokenIds(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = set(stop_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_token_ids:
            return False
        return input_ids[0, -1].item() in self.stop_token_ids


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "你是一个大模型部署工程师。"},
    {"role": "user", "content": "用要点解释：为什么 streaming 能改善体验？"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generation_config = GenerationConfig(
    max_new_tokens=160,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.05,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

stopping_criteria = StoppingCriteriaList(
    [StopOnTokenIds([tokenizer.eos_token_id])]
)

model.generate(
    **inputs,
    generation_config=generation_config,
    streamer=streamer,
    stopping_criteria=stopping_criteria,
)
