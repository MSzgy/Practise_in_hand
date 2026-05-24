# 从 Hugging Face 基础代码理解文本大模型推理：以 Qwen2-0.5B 为例

这篇文章整理自一个文本大模型部署学习项目的基础部分。目标是先把 Hugging Face 文本生成链路讲清楚：一段用户输入如何变成 token，token 如何进入模型，模型如何输出 logits，最后又如何通过解码变成自然语言。

项目中的基础内容按下面这条线展开：

```text
00_pipeline_auto
-> 01_auto_tokenizer_model
-> 02_split_configuration
-> 03_streaming_and_stopping
-> 04_forward_logits_kv_cache
-> 05_architecture_and_intermediates
```

这条顺序很适合学习，因为它从最简单的 `pipeline` 自动封装开始，再逐步拆开 tokenizer、model、generation config、streamer、logits、hidden states 和 KV cache。真正理解这些组件之后，就能对文本模型推理建立稳定的基础框架。

本文以 `Qwen/Qwen2.5-0.5B-Instruct` 这个小模型为例说明。为了表述方便，下面简称 Qwen2-0.5B。这个模型参数量不大，适合在 Notebook 或单卡环境里快速验证文本生成链路。

## 1. 文本生成的最短路径：pipeline 自动挡

Hugging Face 里最简单的文本生成方式是 `pipeline`：

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "用三句话解释：大模型部署里的 tokenizer 是什么？"}
]

result = pipe(
    messages,
    max_new_tokens=120,
    do_sample=False,
)
```

这段代码看起来很短，但背后做了很多事情：

- 根据模型名下载配置文件和权重。
- 自动选择 tokenizer。
- 自动选择模型类，例如 `AutoModelForCausalLM`。
- 根据 chat template 把对话消息转成模型能理解的 prompt。
- 把文本切成 token ids。
- 调用 `model.generate()` 生成新 token。
- 把输出 token ids 解码回文本。

`pipeline` 的好处是快，适合验证模型是否能跑通。但它也隐藏了太多细节。部署时如果只停留在 `pipeline`，就很难解释显存为什么上涨、为什么首 token 慢、为什么流式输出能改善体验、为什么长上下文会吃掉大量 KV cache。

所以基础学习的第一步可以用 `pipeline`，但不能止步于 `pipeline`。

## 2. 拆开 tokenizer 和 model

更接近真实部署的写法，是显式加载 tokenizer 和模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)
```

这里有两个核心对象：

`AutoTokenizer` 负责文本和 token ids 之间的转换。大模型并不直接处理汉字、英文单词或标点，而是处理整数形式的 token id。tokenizer 会把输入文本拆成 token，并把每个 token 映射到词表中的编号。

`AutoModelForCausalLM` 负责加载因果语言模型。Causal LM 的意思是：模型只能根据当前位置之前的 token 来预测下一个 token。GPT、LLaMA、Qwen 这类 decoder-only 模型都属于这一类。

对话模型还需要 chat template：

```python
messages = [
    {"role": "system", "content": "你是一个讲解大模型部署的老师。"},
    {"role": "user", "content": "用三句话解释什么是 chat template。"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

`messages` 是人类容易理解的结构化对话；模型实际训练时看到的通常是某种特殊格式的纯文本。chat template 的作用，就是把 `system/user/assistant` 这样的角色消息转换成模型训练时熟悉的 prompt 格式。

然后再 tokenize：

```python
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
```

这一步通常会得到：

- `input_ids`：token 编号，形状一般是 `[batch_size, seq_len]`。
- `attention_mask`：哪些位置是真实 token，哪些位置是 padding。

最后调用生成：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=120,
    do_sample=False,
)

new_token_ids = outputs[0][inputs["input_ids"].shape[-1]:]
answer = tokenizer.decode(new_token_ids, skip_special_tokens=True)
```

这里的切片很重要：`generate()` 返回的通常是“原始输入 token + 新生成 token”。如果只想拿模型新生成的回答，就需要把 prompt 对应的 token 截掉。

## 3. 把配置拆开看

部署时不能只关心能不能生成，还要关心模型结构、生成策略、设备放置和 dtype。项目基础代码里把配置拆成了几类：

| 模块 | Hugging Face 对象 | 作用 |
| --- | --- | --- |
| 架构配置 | `AutoConfig` | 读取层数、hidden size、attention heads、词表大小等 |
| 分词器 | `AutoTokenizer` | 文本和 token ids 转换，处理 chat template |
| 模型 | `AutoModelForCausalLM` | 加载权重，决定 dtype 和 device map |
| 生成配置 | `GenerationConfig` | 控制最大生成长度、采样、温度、top-p 等 |
| 输入张量 | `tokenizer(...)` | 生成 `input_ids` 和 `attention_mask` |

例如：

```python
from transformers import AutoConfig, GenerationConfig

config = AutoConfig.from_pretrained(model_id)

print("model_type:", config.model_type)
print("hidden_size:", config.hidden_size)
print("num_hidden_layers:", config.num_hidden_layers)
print("num_attention_heads:", config.num_attention_heads)
print("num_key_value_heads:", config.num_key_value_heads)

generation_config = GenerationConfig.from_pretrained(model_id)
generation_config.max_new_tokens = 120
generation_config.do_sample = True
generation_config.temperature = 0.7
generation_config.top_p = 0.9
```

这些参数决定了模型推理时的很多行为：

- `hidden_size` 决定每个 token 在模型内部的向量维度。
- `num_hidden_layers` 决定 Transformer block 的层数。
- `num_attention_heads` 决定 query heads 的数量。
- `num_key_value_heads` 决定 key/value heads 的数量。
- `max_new_tokens` 决定最多生成多少个新 token。
- `temperature` 和 `top_p` 控制采样随机性。
- `do_sample=False` 通常表示贪心生成，输出更稳定。

面试或讲解时，可以把 Hugging Face 文本生成概括成一句话：

> `pipeline` 是高层封装；生产部署中通常要拆开 tokenizer、model、generation config、device、dtype、streaming 和 stopping，这样才能控制显存、延迟、采样策略和输出边界。

## 4. 模型参数怎么看：以 Qwen2-0.5B 的 QKV 维度为例

理解模型参数时，最容易混淆的是 attention 里的 Q、K、V。

Qwen2-0.5B 的典型结构参数如下：

```text
hidden_size = 896
num_hidden_layers = 24
num_attention_heads = 14
num_key_value_heads = 2
vocab_size = 151936
```

每个 attention head 的维度是：

```text
head_dim = hidden_size / num_attention_heads
         = 896 / 14
         = 64
```

假设输入 batch size 为 1，prompt 长度为 32，那么进入某一层 Transformer block 的 hidden states 形状是：

```text
hidden_states: [batch_size, seq_len, hidden_size]
             = [1, 32, 896]
```

### Query 的维度

Qwen2-0.5B 有 14 个 query heads，每个 head 维度 64，所以 query 投影后的总维度仍然是：

```text
num_attention_heads * head_dim
= 14 * 64
= 896
```

因此：

```text
q_proj 输出: [1, 32, 896]
reshape 后: [1, 14, 32, 64]
```

可以理解为：每个 token 会产生 14 组 query，每组 query 是 64 维。

### Key 和 Value 的维度

Qwen2-0.5B 的 `num_key_value_heads` 是 2，而不是 14。这说明它使用的是 grouped-query attention，也就是多个 query heads 共享较少的 key/value heads。

Key 的总投影维度是：

```text
num_key_value_heads * head_dim
= 2 * 64
= 128
```

Value 也是一样：

```text
k_proj 输出: [1, 32, 128]
reshape 后: [1, 2, 32, 64]

v_proj 输出: [1, 32, 128]
reshape 后: [1, 2, 32, 64]
```

因为 query heads 有 14 个，KV heads 只有 2 个，所以平均每 7 个 query heads 共享 1 个 key/value head：

```text
14 query heads / 2 KV heads = 7
```

这就是 GQA 的核心思想：保留较多 query heads 的表达能力，同时减少 key/value 的数量，从而降低 KV cache 的显存占用。

### QKV 权重矩阵可以怎么理解

在 PyTorch 的 `Linear` 层中，权重形状通常是 `[out_features, in_features]`。对 Qwen2-0.5B 来说，某一层 attention 的投影大致可以这样理解：

```text
q_proj weight: [896, 896]
k_proj weight: [128, 896]
v_proj weight: [128, 896]
o_proj weight: [896, 896]
```

含义是：

- `q_proj` 把 896 维 hidden state 投影成 896 维 query。
- `k_proj` 把 896 维 hidden state 投影成 128 维 key。
- `v_proj` 把 896 维 hidden state 投影成 128 维 value。
- `o_proj` 把多头 attention 的结果再投影回 896 维 hidden size。

如果是普通 multi-head attention，Q、K、V 的 head 数通常一样，K/V 也会有 14 个 heads。但 Qwen2-0.5B 这里 K/V 只有 2 个 heads，所以 KV cache 会更省。

## 5. 流式输出：改善的是用户感知延迟

项目基础代码里使用了 `TextStreamer`：

```python
from transformers import TextStreamer

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True,
)

model.generate(
    **inputs,
    generation_config=generation_config,
    streamer=streamer,
)
```

普通生成通常是模型生成完整回答后再一次性返回。流式输出则是在模型每生成一小段内容后，就立刻把它展示给用户。

需要注意的是，streaming 并不会减少模型总计算量。模型还是要一个 token 一个 token 地生成，attention、MLP、logits 计算都不会消失。它真正改善的是用户感知延迟：

- 用户不用等完整答案生成完才看到内容。
- 首 token 出来后，界面就开始响应。
- 长回答场景下，体验会明显更自然。
- 服务端也可以边生成边发送，避免长时间无响应。

基础代码中还演示了 `StoppingCriteria`：

```python
from transformers import StoppingCriteria

class StopOnTokenIds(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = set(stop_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        if not self.stop_token_ids:
            return False
        return input_ids[0, -1].item() in self.stop_token_ids
```

停止条件决定模型什么时候结束生成。最常见的是遇到 `eos_token_id`，也可以根据业务自定义停止 token 或停止字符串。

## 6. 不调用 generate，直接看一次 forward

要理解文本生成的本质，不能只看 `generate()`，还要看一次模型 forward：

```python
import torch

with torch.no_grad():
    outputs = model(**inputs, use_cache=True)

logits = outputs.logits[:, -1, :]
next_token_id = logits.argmax(dim=-1)
next_token = tokenizer.decode(next_token_id)
```

这段代码展示了生成一个 token 的核心过程：

```text
输入 token ids
-> embedding
-> 多层 Transformer block
-> logits
-> 取最后一个位置的 logits
-> 选择下一个 token id
-> decode 成文本
```

`outputs.logits` 的形状通常是：

```text
[batch_size, seq_len, vocab_size]
```

以 Qwen2-0.5B 为例，如果 batch size 是 1，输入长度是 32，词表大小是 151936，那么：

```text
outputs.logits: [1, 32, 151936]
```

为什么只取最后一个位置？

因为 causal LM 的任务是根据前面的上下文预测下一个 token。输入序列中每个位置都会有一份对“下一个 token”的预测，但真正用于继续生成的是最后一个 token 位置的预测结果：

```python
last_token_logits = outputs.logits[:, -1, :]
```

它的形状是：

```text
[batch_size, vocab_size] = [1, 151936]
```

如果使用贪心解码，就是从这 151936 个词表分数中选分数最高的 token：

```python
next_token_id = last_token_logits.argmax(dim=-1)
```

如果使用采样，则会结合 temperature、top-k、top-p 等策略，从概率分布中抽样。

## 7. hidden states、attentions 和中间态

基础代码的最后一部分会打印模型结构和中间态：

```python
with torch.no_grad():
    outputs = model(
        **inputs,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
```

常见输出包括：

```text
logits
hidden_states
attentions
past_key_values
```

`hidden_states` 可以理解为每一层对 token 的内部表示。形状通常是：

```text
每层 hidden state: [batch_size, seq_len, hidden_size]
```

以 Qwen2-0.5B 为例：

```text
[1, 32, 896]
```

如果打开 `output_hidden_states=True`，通常会得到 embedding 输出加上每一层 Transformer block 的输出。Qwen2-0.5B 有 24 层，因此 hidden states 的数量通常是 25 份：1 份 embedding 输出，加 24 份每层输出。

`attentions` 是注意力分数，形状一般类似：

```text
每层 attention: [batch_size, num_heads, seq_len, seq_len]
```

对于长上下文来说，attention map 非常大。例如 seq_len 从 32 增加到 4096，`seq_len * seq_len` 会急剧变大。所以部署中一般不会打开 `output_attentions=True`，除非是调试或分析。

## 8. KV cache 是什么，为什么部署时很重要

自回归生成是一个 token 一个 token 往后生成的。如果每生成一个新 token，都把整个 prompt 和已经生成的内容重新算一遍，那么会非常浪费。

KV cache 的作用是缓存历史 token 在每一层 attention 中的 key 和 value。下一步生成时，新 token 只需要产生自己的 query，并和历史缓存的 key/value 做 attention，不需要重复计算历史 token 的 K/V。

在 Hugging Face 中可以这样打开：

```python
outputs = model(**inputs, use_cache=True)
cache = outputs.past_key_values
```

对 Qwen2-0.5B 来说，某一层的 KV cache 形状可以按下面理解：

```text
key cache:   [batch_size, num_key_value_heads, seq_len, head_dim]
value cache: [batch_size, num_key_value_heads, seq_len, head_dim]
```

带入参数：

```text
key cache:   [1, 2, 32, 64]
value cache: [1, 2, 32, 64]
```

每一层都有一份 key cache 和一份 value cache。Qwen2-0.5B 有 24 层，所以完整 KV cache 会随着层数、batch size、上下文长度一起增长。

一个粗略估算公式是：

```text
KV cache 显存
= 2 * num_hidden_layers * batch_size * num_key_value_heads * seq_len * head_dim * bytes_per_element
```

其中开头的 `2` 表示 key 和 value 两份缓存。

如果使用 bf16 或 fp16，每个元素 2 字节。假设：

```text
num_hidden_layers = 24
batch_size = 1
num_key_value_heads = 2
seq_len = 1024
head_dim = 64
bytes_per_element = 2
```

那么：

```text
KV cache
= 2 * 24 * 1 * 2 * 1024 * 64 * 2
= 12,582,912 bytes
≈ 12 MB
```

这只是 batch size 为 1、上下文长度为 1024 的情况。实际部署中，如果 batch size 增大、并发请求增多、上下文长度变长，KV cache 会成为显存占用的重要来源。

这也是为什么部署工程里经常讨论：

- 最大上下文长度应该设多大。
- 并发请求应该如何调度。
- 为什么长 prompt 会影响吞吐。
- 为什么 GQA/MQA 能降低 KV cache 成本。
- 为什么推理框架要做 KV cache 管理。

## 9. 从输入到输出的完整链路

把基础部分串起来，文本大模型推理可以概括成下面这条链路：

```text
用户消息
-> chat template
-> prompt 文本
-> tokenizer
-> input_ids / attention_mask
-> embedding
-> Transformer blocks
-> hidden states
-> logits
-> decoding strategy
-> next token id
-> KV cache 更新
-> 循环生成
-> tokenizer decode
-> 最终文本
```

对应到 Hugging Face 代码，大致是：

```text
chat template       -> tokenizer.apply_chat_template(...)
tokenizer           -> AutoTokenizer
model config        -> AutoConfig
model weights       -> AutoModelForCausalLM
generation strategy -> GenerationConfig
streaming           -> TextStreamer / TextIteratorStreamer
stop rule           -> eos_token_id / StoppingCriteria
forward outputs     -> logits / hidden_states / past_key_values
KV cache            -> use_cache=True
```

这条链路能帮助我们解释很多基础推理现象。比如：

- 研究显存占用，需要理解模型权重和 KV cache。
- 研究输出稳定性，需要理解 decoding strategy。
- 研究接口体验，需要理解 streaming。

## 10. 基础阶段应该掌握到什么程度

学完这一部分，不一定要马上写推理框架，但至少应该能解释清楚下面几个问题：

1. `pipeline` 帮我们隐藏了哪些步骤？
2. tokenizer、chat template、model 分别负责什么？
3. `AutoModelForCausalLM` 为什么适合 Qwen/GPT/LLaMA 这类模型？
4. `input_ids`、`attention_mask`、`logits` 的形状分别是什么？
5. 为什么生成时只取最后一个位置的 logits？
6. `do_sample=False` 和 temperature/top-p 采样有什么区别？
7. streaming 为什么能改善体验，但不减少总计算量？
8. KV cache 缓存的是什么？
9. Qwen2-0.5B 中 Q、K、V 的维度为什么不一样？
10. 为什么 `num_key_value_heads` 小于 `num_attention_heads` 可以节省显存？

如果这些问题能讲清楚，就说明已经不只是“会调用模型”，而是开始理解文本大模型推理的底层过程了。

## 总结

文本大模型部署的基础，不是先背各种推理框架名字，而是先理解一次生成到底发生了什么。

`pipeline` 让我们快速跑通模型；`AutoTokenizer` 和 `AutoModelForCausalLM` 让我们拆开输入处理和模型推理；`GenerationConfig` 让我们控制生成策略；`TextStreamer` 让输出可以流式返回；直接 forward 则让我们看到 logits、hidden states 和 KV cache。

以 Qwen2-0.5B 为例，模型内部 hidden size 是 896，query heads 是 14，KV heads 是 2，head dim 是 64。因此 Q 的形状可以理解为 `[batch, 14, seq_len, 64]`，K/V 的形状可以理解为 `[batch, 2, seq_len, 64]`。这个维度差异正是 GQA 的体现，也是理解 KV cache 显存优化的入口。

把这些基础概念打牢之后，后面再继续学习更复杂的推理优化，才不会只停留在工具调用层面。
