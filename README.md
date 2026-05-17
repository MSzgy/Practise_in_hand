---
title: Practise In Hand
emoji: 🏢
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.14.0
python_version: '3.12'
app_file: app.py
pinned: false
---

# 大模型部署学习仓库

这个仓库用于系统学习大模型部署：先从 Gradio / Hugging Face Spaces 的最小 Demo 开始，再逐步扩展到文本、音频、图像、视频、多模态和生产化推理服务。

## 当前实验

- [app.py](./app.py): Gradio 文本部署实验台，默认使用 `dphn/dolphin-2.9.4-llama3.1-8b`，可以边看理论边运行 Hugging Face 文本模型实验。
- [text/README.md](./text/README.md): 文本模型 Hugging Face 代码讲解，从 `pipeline` 自动配置逐步拆到 tokenizer、model、generation config、streamer 和 KV cache。

## 面试向理论笔记

- [notes/deployment_interview_theory.md](./notes/deployment_interview_theory.md): 大模型部署核心理论、文本/音频/视频部署链路、显存估算、性能指标、常见面试问答。

## 学习主线

1. Demo 部署：Gradio、Spaces、ZeroGPU、依赖管理。
2. 文本模型部署：tokenizer、chat template、streaming、vLLM、TGI、OpenAI-compatible API。
3. 音频模型部署：ASR、TTS、采样率、切片、实时因子。
4. 图像和视频部署：图片理解、视频抽帧、视频生成、异步任务队列。
5. 生产化部署：Docker、GPU 显存估算、批处理、缓存、监控、压测、成本优化。

## 官方参考

- Hugging Face Spaces: https://huggingface.co/docs/hub/main/spaces
- Hugging Face ZeroGPU: https://huggingface.co/docs/hub/main/spaces-zerogpu
- Gradio: https://www.gradio.app/guides/quickstart
- Text Generation Inference: https://huggingface.co/docs/text-generation-inference/main/en/index
- vLLM OpenAI-compatible Server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
