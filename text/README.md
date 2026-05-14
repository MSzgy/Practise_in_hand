# 文本模型 Hugging Face 代码讲解

这一部分从最简单的“自动挡”开始，逐步拆开文本模型部署中的组件。建议按文件顺序看：

```text
00_pipeline_auto.py
-> 01_auto_tokenizer_model.py
-> 02_split_configuration.py
-> 03_streaming_and_stopping.py
-> 04_forward_logits_kv_cache.py
```

默认模型使用 `HuggingFaceTB/SmolLM2-135M-Instruct`，它足够小，适合学习 Hugging Face Transformers 的调用方式。它不是为了回答质量最强，而是为了让你能在普通环境里更容易跑通链路。

## 0. 最简单：pipeline 自动配置

[00_pipeline_auto.py](./00_pipeline_auto.py) 使用：

```python
from transformers import pipeline

pipe = pipeline("text-generation", model=MODEL_ID)
pipe(messages)
```

`pipeline` 会自动做很多事：

- 下载模型配置。
- 选择合适的 tokenizer。
- 选择合适的 model class。
- 做 chat template。
- tokenize 输入。
- 调用 `model.generate()`。
- decode 输出。

适合快速验证模型，但面试时不能只停在这一层，因为它隐藏了大部分部署细节。

## 1. 拆开 tokenizer 和 model

[01_auto_tokenizer_model.py](./01_auto_tokenizer_model.py) 显式使用：

```python
AutoTokenizer.from_pretrained(...)
AutoModelForCausalLM.from_pretrained(...)
tokenizer.apply_chat_template(...)
model.generate(...)
```

这一层要理解：

- tokenizer 负责把文本变成 token ids。
- chat template 负责把 `system/user/assistant` 消息渲染成模型训练时的格式。
- `AutoModelForCausalLM` 表示加载一个“因果语言模型”，也就是 GPT/LLaMA/Qwen 这类根据上文预测下一个 token 的模型。
- `generate()` 负责循环生成 token。

## 2. 每个模块分开配置

[02_split_configuration.py](./02_split_configuration.py) 把配置拆成几类：

| 模块 | Hugging Face 对象 | 负责什么 |
| --- | --- | --- |
| 架构配置 | `AutoConfig` | 层数、hidden size、attention heads、模型类型 |
| 分词器 | `AutoTokenizer` | tokenization、chat template、pad/eos token |
| 模型 | `AutoModelForCausalLM` | 权重加载、dtype、device map |
| 生成配置 | `GenerationConfig` | max_new_tokens、temperature、top_p、do_sample |
| 输入张量 | `tokenizer(...)` | input_ids、attention_mask |

面试里可以这样说：

> Hugging Face 的 `pipeline` 是上层封装。生产部署时我通常会把 tokenizer、model、generation config、device/dtype、streaming 和 stopping 拆开配置，这样才能控制显存、延迟、采样策略和输出边界。

## 3. 流式输出和停止条件

[03_streaming_and_stopping.py](./03_streaming_and_stopping.py) 加了：

- `TextStreamer`: 生成一个 token 就打印一段文本。
- `StoppingCriteria`: 自定义停止逻辑。
- `GenerationConfig`: 统一管理采样参数。

部署里 streaming 很常见。它通常不减少总计算量，但能降低用户感知的等待时间，因为用户不用等完整答案生成完。

## 4. 看一次 forward：logits 和 KV cache

[04_forward_logits_kv_cache.py](./04_forward_logits_kv_cache.py) 不调用 `generate()`，而是直接：

```python
outputs = model(**inputs, use_cache=True)
logits = outputs.logits[:, -1, :]
next_token_id = logits.argmax(dim=-1)
```

这能看到文本生成的本质：

```text
输入 token ids
-> model forward
-> 最后一个位置的 logits
-> 选出下一个 token
-> decode 成文本
```

`outputs.past_key_values` 就是 KV cache。它是部署面试中的高频点：长上下文和高并发会让 KV cache 变大，从而影响显存和吞吐。

## 推荐运行方式

```bash
pip install -r requirements.txt
python text/00_pipeline_auto.py
python text/01_auto_tokenizer_model.py
python text/02_split_configuration.py
python text/03_streaming_and_stopping.py
python text/04_forward_logits_kv_cache.py
```

第一次运行会下载模型。没有 GPU 也可以学习这些例子，只是速度会慢一些。

## 面试记忆线

文本模型部署可以按这条线讲：

```text
原始文本
-> chat template
-> tokenizer
-> input_ids / attention_mask
-> embedding
-> transformer blocks
-> logits
-> decoding strategy
-> output token ids
-> decode
-> 输出文本
```

Hugging Face 代码中的对应关系：

```text
chat template       -> tokenizer.apply_chat_template(...)
tokenizer           -> AutoTokenizer
model architecture  -> AutoConfig
model weights       -> AutoModelForCausalLM
generation strategy -> GenerationConfig / generate kwargs
streaming           -> TextStreamer / TextIteratorStreamer
stop rule           -> eos_token_id / StoppingCriteria
KV cache            -> use_cache=True / past_key_values
```
