# 语音模型 TTS 学习教程

这一部分先用一个 ModelScope 上可下载运行的小型 TTS 模型理解语音合成链路，再切到参数更大的 VoxCPM2 观察现代多语言、音色设计和克隆模型的部署差异。

```text
00_tts_small_and_voxcpm2.ipynb / 00_tts_small_and_voxcpm2.py
```

## 模型选择

| 层级 | 模型 | 用途 | 重点观察 |
| --- | --- | --- | --- |
| 小模型 | `damo/speech_sambert-hifigan_tts_zh-cn_16k` | 快速理解 TTS 基础链路 | 文本输入、ModelScope pipeline、Sambert 声学模型、HiFiGAN vocoder、16kHz WAV |
| 大模型 | `OpenBMB/VoxCPM2` | 体验更接近真实应用的多语言 TTS | 2B 参数、48kHz 输出、voice design、reference audio cloning、CFG 和 diffusion steps |

小模型默认从 ModelScope 下载，适合在大陆网络环境里先把 TTS 链路跑通。VoxCPM2 也先通过 ModelScope 下载到本地缓存，再从本地目录加载，重点看现代 TTS 部署要点：显存、冷启动、采样率、参考音频质量、长文本切分和实时因子。

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

脚本版默认只运行小模型：

```bash
python speech/00_tts_small_and_voxcpm2.py
```

如果要运行 VoxCPM2：

```bash
RUN_VOXCPM2=1 python speech/00_tts_small_and_voxcpm2.py
```

第一次运行会下载模型权重。VoxCPM2 权重较大，建议在有 GPU 或 Apple Silicon MPS 的环境里运行；CPU 也可以用于理解代码，但速度会明显变慢。

如果要替换模型，可以用环境变量指定 ModelScope 模型 ID：

```bash
SMALL_TTS_MODEL_ID=damo/speech_sambert-hifigan_tts_zh-cn_16k python speech/00_tts_small_and_voxcpm2.py
VOXCPM2_MODEL_ID=OpenBMB/VoxCPM2 RUN_VOXCPM2=1 python speech/00_tts_small_and_voxcpm2.py
```

## 学习主线

语音合成可以按这条线理解：

```text
文本
-> 文本规范化 / tokenizer / processor
-> 说话人条件 speaker embedding 或参考音频
-> 声学模型生成 mel / latent / acoustic representation
-> vocoder / decoder 生成 waveform
-> 保存 WAV / 播放 / 评估时延和质量
```

部署时重点看：

- 输出采样率：16kHz、24kHz、48kHz 会直接影响音质、文件大小和后处理成本。
- 实时因子 RTF：生成耗时 / 音频时长，小于 1 才可能实时。
- 长文本切分：长输入更容易慢、爆显存或出现音色漂移。
- 参考音频质量：克隆场景里，干净、5 到 30 秒的参考音频通常更稳定。
- 模型边界：小模型适合教学和轻量实验，大模型适合质量、控制和多语言能力验证。
