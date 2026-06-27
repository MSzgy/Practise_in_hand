# 语音模型 TTS 学习教程

这一部分用 ModelScope 下载小型 TTS 模型，再用 Transformers 直接加载模型对象，重点不是只跑通 `pipeline`，而是能看到 TTS 模型内部结构。

```text
00_tts_small_and_voxcpm2.ipynb
```

## 模型选择

| 层级 | 模型 | 用途 | 重点观察 |
| --- | --- | --- | --- |
| 小模型 | `microsoft/speecht5_tts` | 跑通 TTS 基础链路 | processor、encoder、decoder、speaker embedding、postnet |
| vocoder | `microsoft/speecht5_hifigan` | 把声学表示还原成 waveform | mel / acoustic representation 到 16kHz WAV |
| 大模型对比 | `OpenBMB/VoxCPM2` | 后续体验现代多语言和音色控制 | 2B 参数、48kHz、voice design、reference audio cloning |

默认 notebook 先使用 `microsoft/speecht5_tts`。它比 VoxCPM2 小很多，并且可以通过 `SpeechT5ForTextToSpeech` 直接查看内部模块，适合教学和面试复盘。这个默认模型更适合英文示例；如果你换成中文 TTS 模型，再把 notebook 里的 `SYNTH_TEXT` 改成中文。

## 推荐运行方式

先安装基础依赖：

```bash
pip install -r requirements.txt
pip install -r speech/requirements-speech.txt
```

然后打开 notebook：

```text
speech/00_tts_small_and_voxcpm2.ipynb
```

如果要替换 ModelScope 模型 ID：

```bash
SMALL_TTS_MODEL_ID=microsoft/speecht5_tts VOCODER_MODEL_ID=microsoft/speecht5_hifigan jupyter notebook
```

## 学习主线

语音合成可以按这条线理解：

```text
文本
-> processor / tokenizer 转成 input_ids
-> speaker embedding 指定谁在说
-> SpeechT5 encoder-decoder 生成声学表示
-> HiFiGAN vocoder 生成 waveform
-> 保存 WAV / 计算 duration 和 RTF
```

notebook 会引导你依次查看：

- `model.config`: 隐藏层维度、encoder/decoder 层数、speaker embedding 维度。
- `model.named_children()`: 顶层模块和参数量分布。
- `model.named_modules()`: 按关键词定位 encoder、decoder、attention、postnet。
- forward hook: 运行时记录关键模块输入输出 shape。
- WAV 指标：采样率、音频时长、生成耗时和 RTF。

部署时重点看：

- 输出采样率：16kHz、24kHz、48kHz 会直接影响音质、文件大小和后处理成本。
- 实时因子 RTF：生成耗时 / 音频时长，小于 1 才可能实时。
- speaker embedding：真实项目里会影响音色稳定性和克隆效果。
- 长文本切分：长输入更容易慢、爆显存或出现音色漂移。
- 模型边界：小模型适合结构学习和轻量实验，大模型适合质量、控制和多语言能力验证。
