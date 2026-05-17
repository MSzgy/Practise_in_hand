from transformers import pipeline


MODEL_ID = "dphn/dolphin-2.9.4-llama3.1-8b"


def print_pipeline_answer(result):
    generated_text = result[0]["generated_text"]

    if isinstance(generated_text, list):
        print(generated_text[-1]["content"])
    else:
        print(generated_text)


pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": "用三句话解释：大模型部署里的 tokenizer 是什么？",
    }
]

result = pipe(
    messages,
    max_new_tokens=120,
    do_sample=False,
)

print_pipeline_answer(result)
