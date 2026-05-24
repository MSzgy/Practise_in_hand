# 大模型原理练习路线

这个目录专门补“模型原理”练习，和 `text/` 的 Hugging Face 调用链、`advanced/` 的部署高级专题、`deployment/` 的框架选型区分开。

## 当前缺口

现有 notebook 已覆盖 tokenizer、chat template、generate、logits、KV cache、模型结构、GQA/RoPE、LoRA、量化、RAG 和部署指标。还需要专门练这些底层原理：

- next-token prediction 到底怎么训练：label shift、teacher forcing、cross entropy、perplexity、padding mask。
- decoder-only Transformer block 怎么从张量一步步拼起来：embedding、causal self-attention、multi-head、residual、RMSNorm/LayerNorm、SwiGLU/MLP、lm head。
- SFT 之后为什么还要偏好对齐：reward model、RLHF、DPO、KL 约束、chosen/rejected 样本。
- 评测和泛化：训练/验证 loss、困惑度、过拟合、数据泄漏、benchmark contamination、校准和人工评测。
- 架构家族差异：encoder-only、decoder-only、encoder-decoder、MLM/CLM/Seq2Seq、MoE、滑动窗口长上下文。

## 建议顺序

```text
00_next_token_loss_perplexity.ipynb
-> 01_transformer_decoder_block_from_scratch.ipynb
-> 02_alignment_dpo_evaluation_principles.ipynb
-> 03_architecture_families_moe_long_context.ipynb
```

## 面试记忆线

```text
文本
-> tokenizer 得到 input_ids
-> 训练时右移 labels 做 next-token prediction
-> embedding + position/RoPE
-> decoder blocks: norm -> attention -> residual -> norm -> MLP -> residual
-> lm_head 得到 logits
-> cross entropy 只监督有效 token
-> perplexity = exp(loss)
-> SFT 学会按指令回答
-> preference alignment 让模型更偏向人类喜欢的答案
-> eval 用 loss、任务指标、人工偏好、安全和线上指标共同判断
-> architecture choice 决定模型适合理解、生成、转换、稀疏专家或长上下文
```

## 面试高频问题

- 为什么 causal LM 训练时输入和 label 都来自同一段文本？
- label shift 是什么？为什么最后一个 token 通常没有 label？
- `ignore_index=-100` 在 SFT 和 padding 里解决什么问题？
- cross entropy、negative log likelihood、perplexity 有什么关系？
- self-attention 里 Q/K/V 分别是什么？causal mask 为什么必须存在？
- RMSNorm 和 LayerNorm 有什么差异？SwiGLU 为什么常见？
- decoder-only、encoder-only、encoder-decoder 的训练目标有什么不同？
- MoE 为什么能扩大参数量但不线性增加每 token 计算量？
- sliding window attention 和完整 attention 的取舍是什么？
- SFT、RLHF、DPO 的区别是什么？DPO 为什么不需要显式训练 reward model？
- 评测一个模型不能只看 demo 的原因是什么？
