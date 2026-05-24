# 模型部署与优化学习路线

这部分专门放部署框架、推理引擎、优化组件和生产化面试点。内容按 2026-05-25 的主流开源生态整理，后续可以继续追加压测、Docker、Kubernetes、监控和多模型路由实验。

## 建议学习顺序

```text
00_deployment_framework_landscape.ipynb
-> 01_optimization_frameworks_and_benchmarking.ipynb
```

## 1. 推理引擎和 LLM serving 框架

这类框架直接负责把模型权重跑起来，并解决 batch 调度、KV cache、流式输出、OpenAI-compatible API、量化和多卡并行等问题。

| 框架 | 适合学习什么 | 典型关键词 |
| --- | --- | --- |
| vLLM | 通用高吞吐 LLM/VLM 服务，生产面试最高频 | PagedAttention、continuous batching、OpenAI API、prefix cache、speculative decoding |
| Hugging Face TGI | Hugging Face 生态里的生产服务 | Docker serving、streaming、metrics、tool/function calling、multi-backend |
| SGLang | 结构化输出、agent/workflow、低延迟服务 | structured generation、router、radix cache、speculative decoding |
| TensorRT-LLM | NVIDIA GPU 上极致性能优化 | engine build、in-flight batching、paged KV cache、FP8/INT8/INT4、tensor parallel |
| LMDeploy | InternLM 生态和 TurboMind/PyTorch engine | TurboMind、continuous batching、量化、多模型服务 |
| llama.cpp | 本地/边缘/CPU/GGUF 部署 | GGUF、CPU/GPU offload、llama-server、OpenAI-compatible API |
| Ollama | 本地模型管理和快速体验 | model library、本地 API、OpenAI compatibility |
| MLC LLM | 编译器路线和跨平台部署 | machine learning compiler、WebGPU、移动端、native deployment |

## 2. 通用模型服务和平台层

这类框架更偏“怎么上线和运维”，经常和上面的 LLM serving engine 组合使用。

| 框架 | 适合学习什么 | 典型关键词 |
| --- | --- | --- |
| NVIDIA Triton Inference Server | 多框架模型服务和企业推理平台 | model repository、dynamic batching、ensemble、HTTP/gRPC、Prometheus |
| Ray Serve | Python 分布式服务和 LLM 应用编排 | autoscaling、replica、deployment graph、OpenAI API、vLLM integration |
| KServe | Kubernetes 上标准化模型服务 | InferenceService、autoscaling、OpenAI-compatible API、AI Gateway |
| BentoML | Python-first API 服务和模型打包 | service、runner、Docker image、adaptive batching、vLLM backend |
| FastAPI + Transformers/vLLM client | 最小服务骨架 | REST/SSE、健康检查、限流、鉴权、异步任务 |

## 3. 优化框架和关键技术

部署面试里不能只背框架名，要能说明瓶颈和优化手段的对应关系。

| 方向 | 代表工具/技术 | 解决什么问题 |
| --- | --- | --- |
| 降精度和量化 | FP16/BF16、FP8、INT8、INT4、bitsandbytes、GPTQ、AWQ、GGUF | 降低权重显存、带宽压力和部署成本 |
| Attention/KV 优化 | FlashAttention、FlashInfer、PagedAttention、paged KV cache | 降低 attention 显存/延迟，提升长上下文和并发能力 |
| Batch 调度 | continuous batching、dynamic batching、in-flight batching、chunked prefill | 提升吞吐，平衡 TTFT 和 TPOT |
| 编译和图优化 | TensorRT-LLM、ONNX Runtime、Optimum、torch.compile、OpenVINO | 减少算子开销，使用硬件专用 kernel |
| 多卡和分布式 | tensor parallel、pipeline parallel、expert parallel、Ray/K8s | 承载更大模型或更高并发 |
| 缓存 | prefix cache、prompt cache、embedding cache、RAG cache | 降低重复请求的 prefill 和检索成本 |
| 解码优化 | speculative decoding、draft model、guided decoding | 降低生成延迟或保证结构化输出 |
| 可观测性 | Prometheus、Grafana、OpenTelemetry、日志采样 | 定位 TTFT、TPOT、排队、OOM、错误率和成本问题 |

## 4. 面试高频主线

一条请求从进入服务到返回结果，可以按这条链路讲：

```text
HTTP/SSE/WebSocket 请求
-> 鉴权/限流/路由
-> tokenizer/chat template
-> scheduler 排队和 batch 合并
-> prefill 建 KV cache
-> decode 循环生成
-> sampling/stopping
-> streaming 返回
-> 指标、日志、trace、计费
```

常见追问：

- vLLM 和普通 Transformers `generate()` 的区别是什么？
- PagedAttention / paged KV cache 为什么能提升并发？
- TTFT、TPOT、吞吐、QPS、并发数之间是什么关系？
- 量化一定会更快吗？为什么有时只省显存不提速？
- continuous batching 和普通静态 batch 有什么区别？
- 为什么长上下文会拖慢 prefill，也会占用更多 KV cache？
- 如果线上 OOM、TTFT 高、TPOT 高、吞吐低，分别怎么排查？
- OpenAI-compatible API 的好处是什么？它不能替代哪些生产能力？

## 官方资料入口

- vLLM: https://docs.vllm.ai
- Hugging Face TGI: https://huggingface.co/docs/text-generation-inference
- NVIDIA TensorRT-LLM: https://nvidia.github.io/TensorRT-LLM
- SGLang: https://docs.sglang.ai
- llama.cpp server: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Ollama OpenAI compatibility: https://docs.ollama.com/openai
- NVIDIA Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
- Ray Serve LLM: https://docs.ray.io/en/latest/serve/llm/index.html
- KServe generative inference: https://kserve.github.io/website/docs/model-serving/generative-inference/overview
- BentoML vLLM serving: https://docs.bentoml.com/en/latest/examples/vllm.html
- FlashInfer: https://docs.flashinfer.ai
- FlashAttention: https://flashattention.org
- Hugging Face Optimum: https://huggingface.co/docs/optimum
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- PyTorch `torch.compile`: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
