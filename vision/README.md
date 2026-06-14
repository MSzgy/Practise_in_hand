# 多模态模型识图教程

这一部分从“图片如何变成模型输入”开始，逐步拆解视觉语言模型的处理链路、内部架构、训练原理和部署成本。

```text
00_how_vlm_sees_images_qwen_gemma4.ipynb
```

## 模型选择

| 层级 | ModelScope 模型 ID | 用途 | 运行建议 |
| --- | --- | --- | --- |
| 小模型 | `Qwen/Qwen2.5-VL-3B-Instruct` | 默认实跑，观察 processor、视觉 token、架构与中间态 | 建议 10GB 以上显存 |
| 大模型 | `google/gemma-4-E4B-it` | 对比新一代原生多模态模型、可变视觉 token budget 和长上下文 | BF16 权重约 16GB，建议 24GB 以上显存 |

Notebook 默认只运行 Qwen2.5-VL-3B。Gemma 4 代码默认关闭，确认显存足够后再设置：

```python
RUN_GEMMA4 = True
```

## 学习主线

```text
图片像素
-> resize / normalize / patchify
-> vision encoder
-> projector / merger
-> 与文本 token embedding 拼接或融合
-> language model
-> 自回归生成文本答案
```

教程重点回答：

1. 模型为什么能够识图，图片进入模型后实际变成了什么张量。
2. Vision Encoder、Projector、Language Model 如何连接。
3. 视觉 token、动态分辨率、位置编码、跨模态对齐和训练阶段。
4. 图片分辨率如何影响 prefill、KV cache、显存、延迟和识别效果。
5. 视觉幻觉、OCR、小目标、计数和提示词偏置等风险如何评估。

## 推荐运行方式

在仓库根目录安装依赖：

```bash
pip install -r requirements.txt
pip install -U -r vision/requirements-vision.txt
```

然后打开：

```text
vision/00_how_vlm_sees_images_qwen_gemma4.ipynb
```

ModelScope Notebook 中默认使用 `modelscope.snapshot_download()` 下载模型。AMD ROCm 环境在 PyTorch 中也会显示为 `cuda:0`，这是正常的兼容接口行为。
