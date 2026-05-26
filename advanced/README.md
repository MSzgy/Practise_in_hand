# 文本大模型高级教程

这一部分面向部署和面试高频主题，建议顺序学习：

```text
10_quantization_memory_speed.ipynb
-> 11_lora_finetuning_sft.ipynb
-> 12_kv_cache_batching_prefill_decode.ipynb
-> 13_interview_drills.ipynb
-> 14_transformer_internals_attention_rope_gqa.ipynb
-> 15_rag_evaluation_serving_patterns.ipynb
-> 16_rag_from_scratch_principles.ipynb
```

## 主题

- [10_quantization_memory_speed.ipynb](./10_quantization_memory_speed.ipynb): dtype 降精度、8bit/4bit 量化、显存估算和生成速度对比。
- [11_lora_finetuning_sft.ipynb](./11_lora_finetuning_sft.ipynb): 用 LoRA 做一个最小 SFT 微调，理解 adapter、可训练参数和保存/加载。
- [12_kv_cache_batching_prefill_decode.ipynb](./12_kv_cache_batching_prefill_decode.ipynb): 拆解 prefill、decode、KV cache、batch padding 和吞吐指标。
- [13_interview_drills.ipynb](./13_interview_drills.ipynb): 文本大模型部署面试高频问题、公式和可运行检查代码。
- [14_transformer_internals_attention_rope_gqa.ipynb](./14_transformer_internals_attention_rope_gqa.ipynb): 用配置和小张量实验解释 hidden size、attention head、MHA/GQA/MQA、RoPE、参数量和 KV cache 公式。
- [15_rag_evaluation_serving_patterns.ipynb](./15_rag_evaluation_serving_patterns.ipynb): 用轻量 TF-IDF 检索练 RAG 流程、chunk/context budget、Recall@k/MRR、引用检查，以及 TTFT/TPOT/吞吐等服务化指标。
- [16_rag_from_scratch_principles.ipynb](./16_rag_from_scratch_principles.ipynb): 不依赖框架或向量库，从原始文档手写 chunk、词袋向量、cosine 检索、context 组装、引用检查和 query rewrite，并映射到常见 RAG 框架，突出底层原理。

## 魔搭 Notebook 建议

先在 Notebook 里选择 GPU，再打开任意 `.ipynb` 按顺序运行。默认模型使用 `Qwen/Qwen2.5-0.5B-Instruct`，用于学习链路；如果显存充足，可以把 `MODEL_ID` 改成 `Qwen/Qwen2.5-7B-Instruct` 做更接近部署的实验。

量化和微调依赖比基础实验更重，所以高级 notebook 会额外安装 `advanced/requirements-advanced.txt`。
