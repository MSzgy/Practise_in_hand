# 大模型部署面试理论笔记

这份笔记偏面试和工程理解，不追求把某一个框架的命令背熟。目标是能讲清楚：模型为什么这样部署、GPU 显存为什么不够、延迟为什么高、吞吐怎么优化、文本/音频/视频部署有什么差异。

## 1. 部署全链路

大模型部署不是只有 `model.generate()`。完整链路通常是：

```text
用户请求
-> 网关 / API Server
-> 参数校验 / 鉴权 / 限流
-> 队列 / 调度
-> tokenizer / processor
-> 推理引擎
-> GPU 执行
-> 后处理
-> streaming 或完整响应
-> 日志 / 监控 / 计费
```

面试里可以把部署分成四层：

1. 应用层：Gradio、FastAPI、Web UI、业务 API。
2. 推理服务层：vLLM、TGI、Triton、Ollama、llama.cpp。
3. 模型运行时：PyTorch、Transformers、CUDA kernel、ONNX Runtime、TensorRT。
4. 基础设施层：GPU、Docker、Kubernetes、对象存储、监控系统。

## 2. 文本大模型核心概念

### Tokenizer

LLM 不直接处理字符串，而是处理 token。部署时 tokenizer 很重要，因为：

- 同一句话在不同 tokenizer 下 token 数不同。
- 输入 token 数影响延迟和显存。
- chat 模型通常需要 chat template，把多轮消息拼成模型训练时见过的格式。
- stop tokens 和特殊 token 配错，可能导致模型不停输出或输出异常。

### 自回归生成

主流文本生成模型通常是 decoder-only autoregressive transformer。它一次预测下一个 token，然后把新 token 拼回上下文继续预测。

所以生成阶段天然有串行部分：

```text
prompt -> token_1 -> token_2 -> token_3 -> ...
```

这也是为什么输出 1000 个 token 通常比输出 100 个 token 慢很多。

### Prefill 和 Decode

推理通常分为两个阶段：

1. Prefill：处理用户输入 prompt，计算第一轮 KV cache。这个阶段可以并行处理整段 prompt，影响首 token 延迟。
2. Decode：逐 token 生成输出。这个阶段每一步依赖上一步，影响输出速度。

常见面试说法：

- 首 token 慢，可能是 prefill、排队、模型冷启动或 prompt 太长。
- 后续 token 慢，通常看 decode 性能、batch 调度、KV cache、GPU 利用率。
- streaming 不会让模型总计算量变少，但能改善用户感知延迟。

## 3. 显存估算

显存主要由几部分组成：

1. 模型权重。
2. KV cache。
3. activation / 临时 buffer。
4. CUDA kernel workspace。
5. 框架额外开销。

### 权重显存

粗略公式：

```text
模型权重显存 ~= 参数量 * 每个参数字节数
```

常见精度：

| 精度 | 每参数字节数 | 说明 |
| --- | ---: | --- |
| FP32 | 4 | 训练常见，推理较少用 |
| FP16 | 2 | GPU 推理常见 |
| BF16 | 2 | 动态范围更好，现代 GPU 常见 |
| FP8 | 1 | 新硬件和新引擎中更常见 |
| INT8 | 1 | 量化推理 |
| INT4 | 0.5 | 更省显存，但质量和速度取决于实现 |

例子：

```text
7B 模型 FP16 权重 ~= 7,000,000,000 * 2 bytes ~= 14 GB
```

但 14GB 只是权重，不包含 KV cache 和运行开销，所以 16GB 显卡不一定稳。

### KV Cache 显存

decoder-only 模型生成时会缓存每层 attention 的 K 和 V，避免重复计算历史 token。

粗略公式：

```text
KV cache ~= batch_size * seq_len * num_layers * 2 * num_kv_heads * head_dim * bytes_per_value
```

其中：

- `2` 表示 K 和 V。
- `seq_len` 是输入加已生成 token 的总长度。
- GQA / MQA 模型的 `num_kv_heads` 会小于 attention heads，所以 KV cache 更省。
- 上下文越长、并发越高，KV cache 增长越明显。

面试重点：

- 权重决定“模型能不能加载”。
- KV cache 决定“长上下文和高并发能不能跑稳”。
- OOM 不一定发生在加载模型时，也可能发生在并发升高或输入变长时。

## 4. 性能指标

部署面试里要区分延迟和吞吐：

| 指标 | 含义 |
| --- | --- |
| TTFT | Time To First Token，首 token 延迟 |
| TPOT | Time Per Output Token，每个输出 token 时间 |
| ITL | Inter-token Latency，token 间隔 |
| E2E Latency | 端到端延迟 |
| Throughput | 吞吐，常用 tokens/s 或 requests/s |
| QPS / RPS | 每秒请求数 |
| p50 / p95 / p99 | 延迟分位数 |
| GPU Utilization | GPU 利用率 |
| Queue Time | 排队时间 |
| Error Rate | 错误率 |

回答性能问题时不要只说“更快”，要说清楚优化的是：

- 首 token 延迟。
- 总响应时间。
- 单请求 token/s。
- 多并发吞吐。
- GPU 成本。
- 稳定性和尾延迟。

## 5. 推理优化技术

### Dynamic Batching

把多个请求合成一个 batch，提高 GPU 利用率。缺点是请求可能需要等待，影响低并发下的延迟。

### Continuous Batching

传统 batching 要等一个 batch 全部完成才能处理下一批。大模型输出长度不同，容易浪费 GPU。

continuous batching 会在生成过程中动态加入新请求、移除完成请求，更适合 LLM 服务。

### PagedAttention / 分页 KV Cache

长文本和多并发时 KV cache 很大。PagedAttention 的核心思路是把 KV cache 像操作系统内存分页一样管理，减少碎片和浪费。vLLM 的高吞吐优势很大程度来自这类 KV cache 管理和调度优化。

### Prefix Cache

如果很多请求共享相同前缀，比如系统提示词、长文档开头，可以复用前缀部分的 KV cache，减少重复 prefill。

### Speculative Decoding

用一个小模型先草拟多个 token，再由大模型验证。理想情况下可以减少大模型 decode 步数，提高速度。难点是实现复杂、收益依赖任务和模型组合。

### Quantization

量化把权重从 FP16/BF16 降到 INT8/INT4 等。优点是省显存，可能提高吞吐。代价是：

- 质量可能下降。
- 某些量化格式需要专门 kernel，不一定更快。
- 长上下文、数学、代码、多语言任务可能更敏感。

## 6. 文本模型部署

文本部署常见路线：

```text
Transformers Demo
-> Gradio / FastAPI
-> vLLM 或 TGI
-> OpenAI-compatible API
-> 网关、鉴权、限流、监控
```

### 关键配置

- `max_model_len`: 最大上下文长度。
- `max_new_tokens`: 单次最多生成 token。
- `temperature`: 随机性。
- `top_p` / `top_k`: 采样范围。
- `stop`: 停止词或停止 token。
- `repetition_penalty`: 重复惩罚。
- `tensor_parallel_size`: 多卡张量并行。
- `gpu_memory_utilization`: 推理引擎可使用的显存比例。

### vLLM 和 TGI 怎么讲

vLLM：

- 适合高吞吐 LLM serving。
- 支持 OpenAI-compatible API。
- 重点能力是 continuous batching、PagedAttention、KV cache 管理、并行推理等。

TGI：

- Hugging Face 官方文本生成推理服务。
- 与 Hugging Face Hub 和 Transformers 生态结合紧密。
- 支持流式输出、张量并行、量化、生产部署相关能力。

面试时不必强行说谁绝对更好，应该说按场景选：

- 已在 Hugging Face 生态里，TGI 很自然。
- 追求 OpenAI API 兼容和高吞吐，vLLM 常见。
- 本地轻量体验，Ollama / llama.cpp 更顺手。

## 7. 音频模型部署

音频部署主要分 ASR 和 TTS。

### ASR: Speech to Text

典型链路：

```text
上传音频
-> 解码格式
-> 重采样
-> 声道转换
-> VAD 或切片
-> ASR 模型
-> 时间戳对齐
-> 文本后处理
```

关键理论：

- 采样率：常见 16kHz、44.1kHz、48kHz。模型通常有固定期望采样率。
- VAD：Voice Activity Detection，用来切掉静音，降低计算量。
- Chunking：长音频要切片，否则显存和延迟不可控。
- WER / CER：语音识别质量指标，分别是词错误率和字错误率。
- RTF：Real Time Factor，处理 10 秒音频耗时 5 秒，则 RTF=0.5，表示快于实时。

面试常问：

- 为什么长音频不能直接整段送模型？
- 如何做流式 ASR？
- 如何合并切片后的时间戳？
- 如何处理背景噪声、多人说话、方言？

### TTS: Text to Speech

典型链路：

```text
输入文本
-> 文本规范化
-> 分句
-> 文本/音素编码
-> 声学模型或生成模型
-> vocoder / waveform 生成
-> 音频后处理
```

关键理论：

- 文本规范化：数字、日期、英文缩写要读对。
- 分句：长文本一次生成容易慢，也容易语气不稳定。
- vocoder：把中间声学表示变成波形。
- 流式 TTS：边生成边播放，重点是降低首包延迟。
- 指标：主观 MOS、延迟、稳定性、音色相似度。

## 8. 图像和视频模型部署

视频部署通常比文本更复杂，因为输入输出体积大、预处理重、任务耗时长。

### 图像理解

常见任务：

- image captioning。
- OCR。
- visual question answering。
- object detection。
- image embedding。

关键点：

- 图片需要 resize、normalize、patch embedding。
- 多图输入会增加显存和 prefill 成本。
- OCR 场景还要关注分辨率、旋转、表格结构。

### 视频理解

典型链路：

```text
上传视频
-> 解码
-> 抽帧
-> 关键帧选择
-> 图像/视频模型推理
-> 时间线聚合
-> 摘要或问答结果
```

关键理论：

- 抽帧策略比盲目全帧输入更重要。
- 视频可以按时间窗口切片，再汇总结果。
- 帧率、分辨率、时长都会影响成本。
- 视频问答通常需要保留时间戳，方便定位证据。

### 视频生成

典型特点：

- 推理时间长。
- 显存占用高。
- 输出文件大。
- 更适合异步任务队列，而不是同步 HTTP 请求。

推荐架构：

```text
提交任务
-> 返回 task_id
-> 后台 GPU worker 执行
-> 对象存储保存结果
-> 前端轮询或 WebSocket 获取状态
```

面试时可以强调：视频生成服务重点不是只把模型跑起来，而是任务调度、队列、失败重试、存储、超时控制和成本控制。

## 9. 多模态部署

多模态模型通常包含不同 processor：

- 文本 tokenizer。
- 图片 processor。
- 音频 feature extractor。
- 视频 frame sampler。

多模态部署难点：

- 输入大小不可控。
- 预处理耗时明显。
- 请求结构复杂。
- 模型上下文中图片 token / 音频 token 会占用窗口。
- 文件上传和安全扫描也属于部署问题。

常见优化：

- 限制文件大小、时长、分辨率。
- 预处理放到 CPU worker。
- GPU 只做核心推理。
- 长任务走异步队列。
- 结果存对象存储，只在 API 中返回 URL 或 task_id。

## 10. 生产化架构

一个相对完整的生产架构：

```text
Client
-> API Gateway
-> Auth / Rate Limit
-> Request Router
-> Queue
-> Inference Workers
-> Model Cache / Object Storage
-> Response Streaming / Task Result
-> Logs / Metrics / Traces
```

需要关注：

- 冷启动：模型加载很慢，应该尽量常驻。
- 扩缩容：GPU 服务扩容慢，不能只靠瞬时扩容。
- 限流：防止超长输入和高并发打爆显存。
- 超时：长任务要异步。
- 降级：模型不可用时返回明确错误或切到小模型。
- 观测：没有日志和指标，很难定位延迟和 OOM。

## 11. 面试高频问答

### 1. 一个 7B FP16 模型需要多少显存？

权重约 14GB，但实际推理需要加上 KV cache、activation、CUDA workspace 和框架开销。长上下文或高并发会显著增加 KV cache，所以不能只按权重判断。

### 2. 为什么 vLLM 通常比直接用 Transformers pipeline 更适合服务化？

因为服务化关注多并发吞吐和显存管理。vLLM 有 continuous batching、PagedAttention、OpenAI-compatible server 等能力，比单请求脚本式推理更适合线上服务。

### 3. 首 token 延迟高怎么排查？

看请求是否排队、prompt 是否过长、prefill 是否慢、模型是否冷启动、tokenizer 或预处理是否耗时、GPU 是否满载。

### 4. 输出 token 慢怎么优化？

可以考虑更快的推理引擎、量化、continuous batching、合适的 batch 策略、缩短输出长度、speculative decoding、多卡并行或更强 GPU。

### 5. Streaming 能减少总推理时间吗？

通常不能。Streaming 主要改善用户感知延迟，让用户更早看到结果，但模型仍然要逐 token 生成。

### 6. 为什么长上下文很贵？

prefill 需要处理更多输入 token，KV cache 也随上下文长度线性增长。长上下文还可能降低吞吐、增加显存碎片和排队时间。

### 7. 量化一定会更快吗？

不一定。量化通常省显存，但速度取决于硬件和 kernel 支持。没有高效 kernel 时，INT4/INT8 可能只是省显存，不一定提升延迟。

### 8. 音频 ASR 部署和文本 LLM 部署最大区别是什么？

ASR 多了音频解码、重采样、VAD、切片和时间戳合并。指标也不同，除了延迟和吞吐，还要看 WER/CER 和 RTF。

### 9. 视频生成为什么常用异步队列？

因为视频生成耗时长、显存占用高、失败概率和重试成本都高。同步 HTTP 容易超时，也不利于任务排队和状态管理。

### 10. 如何判断部署是否生产可用？

至少要看：稳定的 API、明确的超时和限流、日志监控、压测结果、错误处理、模型版本管理、成本估算、数据安全和回滚方案。

## 12. 面试表达模板

回答部署问题时可以按这个顺序：

```text
1. 先说目标：低延迟、高吞吐、低成本还是高质量。
2. 再说模型：参数量、精度、上下文、输入模态。
3. 再说资源：GPU 型号、显存、并发、预期 QPS。
4. 再说服务框架：Gradio / FastAPI / vLLM / TGI / Triton。
5. 再说优化：batching、KV cache、量化、缓存、异步队列。
6. 最后说观测：TTFT、TPOT、p95、GPU 利用率、OOM、错误率。
```

这套结构比直接背框架名更有说服力。

## 13. 推荐继续补的实验

1. 文本：用 Gradio 做一个 streaming chat demo。
2. 文本：用 vLLM 起一个 OpenAI-compatible API，再写 client 调用。
3. 文本：记录 7B 模型在不同 `max_new_tokens` 下的 TTFT 和总耗时。
4. 音频：做 Whisper ASR，比较短音频和长音频切片。
5. 音频：做 TTS，并记录首包延迟和总生成时间。
6. 视频：做抽帧 + 图片理解模型的视频摘要。
7. 视频：做异步任务队列，返回 `task_id` 和任务状态。
8. 生产化：给服务加日志、限流、超时和简单压测脚本。

## 14. 官方参考

- Hugging Face Spaces: https://huggingface.co/docs/hub/main/spaces
- Hugging Face ZeroGPU: https://huggingface.co/docs/hub/main/spaces-zerogpu
- Gradio Quickstart: https://www.gradio.app/guides/quickstart
- Hugging Face Text Generation Inference: https://huggingface.co/docs/text-generation-inference/main/en/index
- vLLM OpenAI-compatible Server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Hugging Face Inference Endpoints: https://huggingface.co/docs/huggingface_hub/main/en/guides/inference_endpoints
