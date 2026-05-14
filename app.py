from functools import lru_cache
from pathlib import Path
from time import perf_counter

import gradio as gr
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
ROOT = Path(__file__).parent
TEXT_DIR = ROOT / "text"
THEORY_PATH = ROOT / "notes" / "deployment_interview_theory.md"

LESSONS = {
    "00 pipeline 自动配置": "00_pipeline_auto.py",
    "01 tokenizer + model": "01_auto_tokenizer_model.py",
    "02 模块分开配置": "02_split_configuration.py",
    "03 streaming + stopping": "03_streaming_and_stopping.py",
    "04 forward + logits + KV cache": "04_forward_logits_kv_cache.py",
}

LESSON_NOTES = {
    "00 pipeline 自动配置": """
`pipeline("text-generation")` 是最高层封装。它会自动加载 tokenizer、model、generation config，并完成 tokenize、generate、decode。

面试表达：pipeline 适合快速验证；生产部署通常需要把 tokenizer、model、生成参数、设备和流式输出拆开控制。
""",
    "01 tokenizer + model": """
这一层显式拆开 `AutoTokenizer` 和 `AutoModelForCausalLM`。

关键链路：messages -> chat template -> token ids -> model.generate -> new token ids -> decode。
""",
    "02 模块分开配置": """
这一层把架构配置、分词器配置、模型加载配置、生成配置分开。

面试表达：拆开配置后才能精确控制 dtype、device、max_new_tokens、temperature、top_p、pad/eos token 等部署关键项。
""",
    "03 streaming + stopping": """
streaming 不是减少总计算量，而是让用户更早看到输出。

停止条件通常来自 `eos_token_id`、stop string、`max_new_tokens` 或自定义 `StoppingCriteria`。
""",
    "04 forward + logits + KV cache": """
不调用 `generate()`，直接看一次 forward。

模型输出 logits，解码策略从 logits 中选下一个 token。`past_key_values` 就是 KV cache，是长上下文和高并发部署里的显存大头之一。
""",
}


def read_text(path, fallback):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


TEXT_README = read_text(TEXT_DIR / "README.md", "文本教程文件还没有生成。")
THEORY_TEXT = read_text(THEORY_PATH, "理论笔记文件还没有生成。")


def device_summary():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"cuda:0 ({name})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_load_kwargs():
    if torch.cuda.is_available():
        return {"torch_dtype": torch.float16, "device_map": "auto"}
    return {"torch_dtype": torch.float32}


@lru_cache(maxsize=2)
def load_components(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        low_cpu_mem_usage=True,
        **model_load_kwargs(),
    )

    if not hasattr(model, "hf_device_map"):
        if torch.cuda.is_available():
            model = model.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")

    model.eval()
    return tokenizer, model, config


def model_device(model):
    return next(model.parameters()).device


def build_messages(system_prompt, user_prompt):
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def build_prompt(tokenizer, messages):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role']}: {message['content']}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def generation_config(tokenizer, max_new_tokens, do_sample, temperature, top_p):
    config = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "repetition_penalty": 1.05,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        config["temperature"] = float(temperature)
        config["top_p"] = float(top_p)

    return GenerationConfig(**config)


def inspect_config(config, tokenizer, model):
    rows = [
        ("model_type", getattr(config, "model_type", "unknown")),
        ("hidden_size", getattr(config, "hidden_size", "unknown")),
        ("num_hidden_layers", getattr(config, "num_hidden_layers", "unknown")),
        ("num_attention_heads", getattr(config, "num_attention_heads", "unknown")),
        ("num_key_value_heads", getattr(config, "num_key_value_heads", "unknown")),
        ("vocab_size", getattr(config, "vocab_size", "unknown")),
        ("pad_token_id", tokenizer.pad_token_id),
        ("eos_token_id", tokenizer.eos_token_id),
        ("runtime_device", str(model_device(model))),
    ]
    return "\n".join(f"{key}: {value}" for key, value in rows)


def explain_tokens(tokenizer, prompt):
    encoded = tokenizer(prompt, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    preview = list(zip(ids[:80], tokens[:80]))
    lines = [f"token_count: {len(ids)}", "", "first_tokens:"]
    lines.extend(f"{token_id:>8}  {token}" for token_id, token in preview)
    if len(ids) > 80:
        lines.append(f"... {len(ids) - 80} more tokens")
    return "\n".join(lines)


def run_experiment(
    lesson_name,
    model_id,
    system_prompt,
    user_prompt,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
):
    if not user_prompt.strip():
        raise gr.Error("请输入 user prompt。")

    start = perf_counter()
    tokenizer, model, config = load_components(model_id.strip() or DEFAULT_MODEL_ID)
    messages = build_messages(system_prompt, user_prompt)
    prompt = build_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    input_len = inputs["input_ids"].shape[-1]

    if lesson_name == "04 forward + logits + KV cache":
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)

        logits = outputs.logits[:, -1, :]
        next_token_id = logits.argmax(dim=-1)
        next_token = tokenizer.decode(next_token_id)
        cache = outputs.past_key_values

        try:
            cache_layers = len(cache)
        except TypeError:
            cache_layers = "unknown"

        if hasattr(cache, "get_seq_length"):
            cache_seq_len = cache.get_seq_length()
        else:
            cache_seq_len = input_len

        elapsed = perf_counter() - start
        answer = (
            "这是一次 forward 观察，不是完整生成。\n\n"
            f"greedy next token id: {next_token_id.item()}\n"
            f"greedy next token: {next_token!r}\n"
            f"logits shape: {tuple(logits.shape)}\n"
            f"KV cache type: {type(cache).__name__}\n"
            f"KV cache layers: {cache_layers}\n"
            f"KV cache sequence length: {cache_seq_len}"
        )
        metrics = {
            "lesson": lesson_name,
            "input_tokens": input_len,
            "new_tokens": 1,
            "elapsed_seconds": round(elapsed, 3),
            "device": str(model_device(model)),
        }
        return answer, inspect_config(config, tokenizer, model), explain_tokens(tokenizer, prompt), metrics

    gen_config = generation_config(
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)

    new_token_ids = outputs[0][input_len:]
    answer = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    elapsed = perf_counter() - start

    metrics = {
        "lesson": lesson_name,
        "input_tokens": input_len,
        "new_tokens": int(new_token_ids.shape[-1]),
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(float(new_token_ids.shape[-1]) / elapsed, 3)
        if elapsed > 0
        else None,
        "device": str(model_device(model)),
    }
    return answer, inspect_config(config, tokenizer, model), explain_tokens(tokenizer, prompt), metrics


def lesson_note(lesson_name):
    return LESSON_NOTES.get(lesson_name, "")


def lesson_code(lesson_name):
    file_name = LESSONS.get(lesson_name)
    if not file_name:
        return ""
    return read_text(TEXT_DIR / file_name, "代码文件不存在。")


def lesson_file_name(lesson_name):
    return LESSONS.get(lesson_name, "")


with gr.Blocks(title="大模型文本部署实验台", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 大模型文本部署实验台

从 Hugging Face `pipeline` 自动配置开始，逐步拆到 tokenizer、model、generation config、streaming、logits 和 KV cache。
"""
    )

    with gr.Tabs():
        with gr.Tab("实验"):
            with gr.Row():
                with gr.Column(scale=4):
                    lesson = gr.Dropdown(
                        choices=list(LESSONS.keys()),
                        value="00 pipeline 自动配置",
                        label="实验层级",
                    )
                    model_id = gr.Textbox(
                        value=DEFAULT_MODEL_ID,
                        label="模型",
                    )
                    system_prompt = gr.Textbox(
                        value="你是一个讲解大模型部署的老师，回答要清晰、简洁、面试友好。",
                        label="system prompt",
                        lines=2,
                    )
                    user_prompt = gr.Textbox(
                        value="用三句话解释 tokenizer 在文本大模型部署中的作用。",
                        label="user prompt",
                        lines=4,
                    )

                with gr.Column(scale=3):
                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=256,
                        value=120,
                        step=8,
                        label="max_new_tokens",
                    )
                    do_sample = gr.Checkbox(value=False, label="do_sample")
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.5,
                        value=0.7,
                        step=0.1,
                        label="temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="top_p",
                    )
                    run_btn = gr.Button("运行", variant="primary")

            with gr.Row():
                lesson_summary = gr.Markdown(value=lesson_note("00 pipeline 自动配置"))
                runtime = gr.Textbox(
                    value=f"Space runtime: {device_summary()}",
                    label="运行环境",
                    interactive=False,
                )

            with gr.Row():
                answer = gr.Textbox(label="模型输出", lines=12)
                metrics = gr.JSON(label="指标")

            with gr.Accordion("token 与配置观察", open=False):
                with gr.Row():
                    config_view = gr.Textbox(label="模型与配置", lines=10)
                    token_view = gr.Textbox(label="token 预览", lines=10)

        with gr.Tab("理论"):
            gr.Markdown(TEXT_README)
            gr.Markdown(THEORY_TEXT)

        with gr.Tab("代码"):
            code_selector = gr.Dropdown(
                choices=list(LESSONS.keys()),
                value="00 pipeline 自动配置",
                label="代码文件",
            )
            code_file = gr.Textbox(
                value=lesson_file_name("00 pipeline 自动配置"),
                label="文件名",
                interactive=False,
            )
            code_view = gr.Code(
                value=lesson_code("00 pipeline 自动配置"),
                language="python",
                label="源码",
            )

    lesson.change(lesson_note, inputs=lesson, outputs=lesson_summary)
    run_btn.click(
        run_experiment,
        inputs=[
            lesson,
            model_id,
            system_prompt,
            user_prompt,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
        ],
        outputs=[answer, config_view, token_view, metrics],
    )
    code_selector.change(lesson_file_name, inputs=code_selector, outputs=code_file)
    code_selector.change(lesson_code, inputs=code_selector, outputs=code_view)


if __name__ == "__main__":
    demo.queue(max_size=16).launch()
