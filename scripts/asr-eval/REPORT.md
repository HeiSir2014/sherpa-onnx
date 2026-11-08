# 本地 ASR 模型评测报告：会议 vs 影视字幕

> 本文件由 `scripts/asr-eval/make_report.py` 根据评测结果自动生成，请勿手工修改；重新评测后运行 `python make_report.py` 即可更新。

## 1. 结论与推荐

> 推荐结论为人工判断；理由中的数字由评测结果自动计算。

| 用途 | 推荐模型 | 理由 |
|---|---|---|
| 会议实时字幕 | **SenseVoice-FunASR-Nano-CTC** + VAD | 会议 CER 10.52%，比第一名（FireRedASR2-AED）只差 1.4 个百分点；RTF 0.033，速度是它的 20 倍 |
| 会后纪要 | **FireRedASR2-AED**（要热词时用 Qwen3-ASR-0.6B） | 会议总 CER 第一（9.08%），5 段中 4 段最佳；sherpa-onnx 里的 FireRed 不支持热词 |
| 影视字幕（质量优先） | **FireRedASR2-AED** | 影视总 CER 第一（14.46%），5 段中 4 段最佳 |
| 影视字幕（成本优先） | **SenseVoice-Small（原版）** | 影视 CER 18.06%，RTF 0.032；CTC 自带真实 token 时间戳，便于对齐字幕时间轴 |
| 不推荐 | Fun-ASR-Nano（LLM 版）、Whisper large-v3-turbo；普通话场景不用两个方言微调版（SenseVoice 粤语版、Paraformer 川渝版） | Whisper 会议/影视 CER 19.8%/36.7%；Fun-ASR-Nano（fp16）影视 CER 24.4%，会议 RTF 1.01；方言微调版不是通用普通话模型 |

## 2. 会议场景总体结果（5 段 × 8 分钟）

| 排名 | 模型 | 精度 | 参数量 | 模型大小 | CER | 替换 | 删除 | 插入 | RTF | 1 小时音频耗时 | 加载时间 | 幻觉（字/分钟） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FireRedASR2-AED | INT8 98% / FP32 2% | 1172M | 1177 MB | **9.08%** | 3.11% | 2.16% | 3.82% | 0.661 | 39.7 分钟 | 5.1 s | 0.0 |
| 2 | Qwen3-ASR-0.6B | INT8 99% / FP32 1% | 939M | 937 MB | **9.62%** | 3.68% | 2.28% | 3.67% | 0.412 | 24.7 分钟 | 4.0 s | 0.0 |
| 3 | SenseVoice-FunASR-Nano-CTC | INT8 100% | 258M | 251 MB | **10.52%** | 4.33% | 2.48% | 3.71% | 0.033 | 2.0 分钟 | 1.5 s | 0.0 |
| 4 | SenseVoice-Small（粤语微调版） | INT8 100% | 232M | 226 MB | **10.95%** | 4.80% | 2.84% | 3.31% | 0.101 | 6.1 分钟 | 1.7 s | 0.0 |
| 5 | SenseVoice-Small（原版） | INT8 100% | 234M | 228 MB | **10.99%** | 4.38% | 2.88% | 3.73% | 0.059 | 3.5 分钟 | 1.5 s | 0.1 |
| 6 | FireRedASR2-CTC | INT8 98% / FP32 2% | 734M | 740 MB | **11.08%** | 3.75% | 4.24% | 3.08% | 0.476 | 28.5 分钟 | 2.4 s | 0.0 |
| 7 | Fun-ASR-Nano-2512（LLM fp16） | FP16 61% / INT8 39% | 983M | 1513 MB | **12.02%** | 3.68% | 4.18% | 4.17% | 1.015 | 60.9 分钟 | 9.2 s | 0.1 |
| 8 | Paraformer-large（川渝方言微调版） | INT8 97% / FP32 3% | 216M | 227 MB | **13.24%** | 5.85% | 2.68% | 4.71% | 0.104 ⚠ | 6.3 分钟 | 3.4 s | 0.0 |
| 9 | Whisper large-v3-turbo | INT8 91% / FP32 9% | 809M | 988 MB | **19.84%** | 8.19% | 8.88% | 2.78% | 1.470 ⚠ | 88.2 分钟 | 3.1 s | 0.0 |
| 10 | Fun-ASR-Nano-2512（全 int8 + 重复惩罚） | INT8 100% | 984M | 948 MB | **66.58%** | 13.86% | 51.54% | 1.19% | 0.312 | 18.7 分钟 | 5.5 s | 0.1 |

逐段 CER：

| 模型 | roundtable-yuanzhuopai | roundtable-xunjie | podcast-caikangyong | interview-rising | talk-yike |
|---|---|---|---|---|---|
| FireRedASR2-AED | **14.0%** | **7.3%** | **7.1%** | **12.3%** | 5.6% |
| Qwen3-ASR-0.6B | 14.6% | 8.3% | 7.2% | 13.8% | **5.2%** |
| SenseVoice-FunASR-Nano-CTC | 16.2% | 9.4% | 8.4% | 13.6% | 6.0% |
| SenseVoice-Small（粤语微调版） | 15.9% | 9.8% | 8.5% | 15.1% | 6.3% |
| SenseVoice-Small（原版） | 15.7% | 9.1% | 8.2% | 15.2% | 7.8% |
| FireRedASR2-CTC | 14.8% | 9.7% | 9.7% | 15.2% | 6.7% |
| Fun-ASR-Nano-2512（LLM fp16） | 17.2% | 11.0% | 9.2% | 15.4% | 8.3% |
| Paraformer-large（川渝方言微调版） | 16.1% | 10.6% | 9.3% | 25.4% | 6.2% |
| Whisper large-v3-turbo | 27.0% | 23.2% | 16.2% | 21.4% | 12.0% |
| Fun-ASR-Nano-2512（全 int8 + 重复惩罚） | 73.8% | 64.7% | 65.9% | 62.9% | 66.5% |

（加粗为该段最佳）

## 3. 影视场景总体结果（5 段 × 8 分钟）

| 排名 | 模型 | 精度 | 参数量 | 模型大小 | CER | 替换 | 删除 | 插入 | RTF | 1 小时音频耗时 | 加载时间 | 幻觉（字/分钟） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FireRedASR2-AED | INT8 98% / FP32 2% | 1172M | 1177 MB | **14.46%** | 4.83% | 5.03% | 4.60% | 0.377 | 22.6 分钟 | 5.1 s | 1.3 |
| 2 | Qwen3-ASR-0.6B | INT8 99% / FP32 1% | 939M | 937 MB | **16.25%** | 6.82% | 4.96% | 4.47% | 0.229 | 13.7 分钟 | 4.8 s | 1.2 |
| 3 | FireRedASR2-CTC | INT8 98% / FP32 2% | 734M | 740 MB | **17.90%** | 7.22% | 7.66% | 3.02% | 0.260 | 15.6 分钟 | 2.4 s | 0.9 |
| 4 | SenseVoice-Small（原版） | INT8 100% | 234M | 228 MB | **18.06%** | 7.49% | 5.94% | 4.63% | 0.032 | 1.9 分钟 | 1.5 s | 1.6 |
| 5 | SenseVoice-FunASR-Nano-CTC | INT8 100% | 258M | 251 MB | **18.75%** | 9.36% | 5.25% | 4.13% | 0.025 | 1.5 分钟 | 1.5 s | 1.2 |
| 6 | SenseVoice-Small（粤语微调版） | INT8 100% | 232M | 226 MB | **19.06%** | 9.27% | 6.75% | 3.04% | 0.050 | 3.0 分钟 | 1.7 s | 1.2 |
| 7 | Paraformer-large（川渝方言微调版） | INT8 97% / FP32 3% | 216M | 227 MB | **21.97%** | 10.95% | 7.46% | 3.55% | 0.046 ⚠ | 2.8 分钟 | 3.4 s | 1.4 |
| 8 | Fun-ASR-Nano-2512（LLM fp16） | FP16 61% / INT8 39% | 983M | 1513 MB | **24.36%** | 6.64% | 11.84% | 5.88% | 0.539 | 32.3 分钟 | 9.2 s | 3.3 |
| 9 | Whisper large-v3-turbo | INT8 91% / FP32 9% | 809M | 988 MB | **36.74%** | 14.48% | 13.45% | 8.80% | 0.570 ⚠ | 34.2 分钟 | 3.1 s | 2.2 |
| 10 | Fun-ASR-Nano-2512（全 int8 + 重复惩罚） | INT8 100% | 984M | 948 MB | **70.50%** | 12.87% | 57.14% | 0.49% | 0.175 | 10.5 分钟 | 5.5 s | 2.7 |

逐段 CER：

| 模型 | drama-cuihuoniandai | costume-zhuyu | scifi-santi | action-nuhuo | comedy-jiongba |
|---|---|---|---|---|---|
| FireRedASR2-AED | 19.9% | **12.3%** | **13.3%** | **15.1%** | **14.3%** |
| Qwen3-ASR-0.6B | **16.1%** | 13.8% | 16.4% | 18.7% | 17.5% |
| FireRedASR2-CTC | 16.5% | 18.2% | 18.2% | 23.4% | 17.0% |
| SenseVoice-Small（原版） | 17.8% | 19.7% | 17.2% | 19.4% | 17.6% |
| SenseVoice-FunASR-Nano-CTC | 17.4% | 18.9% | 19.5% | 21.8% | 17.9% |
| SenseVoice-Small（粤语微调版） | 20.1% | 19.7% | 19.1% | 23.4% | 17.1% |
| Paraformer-large（川渝方言微调版） | 23.0% | 21.0% | 21.7% | 35.3% | 19.8% |
| Fun-ASR-Nano-2512（LLM fp16） | 22.6% | 40.4% | 16.1% | 29.0% | 21.1% |
| Whisper large-v3-turbo | 33.8% | 44.4% | 36.2% | 44.4% | 31.6% |
| Fun-ASR-Nano-2512（全 int8 + 重复惩罚） | 70.2% | 68.5% | 67.8% | 74.2% | 74.3% |

（加粗为该段最佳）

⚠ Whisper large-v3-turbo：会议素材中有 4 段运行时机器在编译，RTF 偏高；Paraformer-large（川渝方言微调版）：部分素材运行时有诊断任务并行，RTF 可能略高。

## 4. 模型清单

| 模型 | 结构 | 语言 | 参数量 | 精度构成 | 文件 | 来源 | sherpa-onnx 模型包 |
|---|---|---|---|---|---|---|---|
| FireRedASR2-AED | Conformer 编码器 + Transformer 注意力解码器（自回归） | 普通话（含 20+ 方言/口音）、英语 | 1171.7M | INT8 98% / FP32 2% | `decoder.int8.onnx` 398 MB INT8<br>`encoder.int8.onnx` 779 MB INT8 | [链接](https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED) | `sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26` |
| Qwen3-ASR-0.6B | 音频编码器 + Qwen3 LLM 解码器（自回归） | 30 种语言 + 22 种方言 | 939.3M | INT8 99% / FP32 1% | `conv_frontend.onnx` 42 MB FP32<br>`decoder.int8.onnx` 721 MB INT8<br>`encoder.int8.onnx` 174 MB INT8 | [链接](https://github.com/QwenLM/Qwen3-ASR) | `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25` |
| FireRedASR2-CTC | FireRedASR2-AED 的 Conformer 编码器 + CTC 分支（非自回归） | 普通话、英语 | 734.4M | INT8 98% / FP32 2% | `model.int8.onnx` 740 MB INT8 | [链接](https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED) | `sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25` |
| SenseVoice-FunASR-Nano-CTC | Fun-ASR-Nano-2512 的音频编码器 + CTC 头（无 LLM，非自回归） | 中文（7 大方言、26 地区口音）、英语、日语等 | 258.1M | INT8 100% | `model.int8.onnx` 251 MB INT8 | [链接](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | `sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17` |
| SenseVoice-Small（原版） | SAN-M 编码器 + CTC（非自回归） | 普通话、粤语、英语、日语、韩语 | 234.0M | INT8 100% | `model.int8.onnx` 228 MB INT8 | [链接](https://www.modelscope.cn/models/iic/SenseVoiceSmall) | `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17` |
| SenseVoice-Small（粤语微调版） | SAN-M 编码器 + CTC（非自回归）；原版再用 2.18 万小时粤语微调 | 粤语为主，兼顾普通话、英语、日语、韩语 | 231.9M | INT8 100% | `model.int8.onnx` 226 MB INT8 | [链接](https://huggingface.co/ASLP-lab/WSYue-ASR) | `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09` |
| Fun-ASR-Nano-2512（LLM fp16） | 音频编码器 + 适配器 + Qwen3-0.6B LLM（自回归） | 中文（7 大方言、26 地区口音）、英语、日语等 31 种 | 983.3M | FP16 61% / INT8 39% | `encoder_adaptor.int8.onnx` 227 MB INT8<br>`embedding.int8.onnx` 148 MB INT8<br>`llm.fp16.onnx` 1138 MB FP16 | [链接](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | `sherpa-onnx-funasr-nano-int8-2025-12-30 的编码器/embedding + ModelScope zengshuishui/FunASR-nano-onnx 的 llm_fp16` |
| Paraformer-large（川渝方言微调版） | Paraformer 非自回归编解码 | 普通话、川渝方言 | 216.4M | INT8 97% / FP32 3% | `model.int8.onnx` 227 MB INT8 | [链接](https://huggingface.co/ASLP-lab/WSChuan-ASR) | `sherpa-onnx-paraformer-zh-int8-2025-10-07` |
| Whisper large-v3-turbo | Transformer 编解码（自回归） | 99 种语言 | 808.9M | INT8 91% / FP32 9% | `turbo-decoder.int8.onnx` 344 MB INT8<br>`turbo-encoder.int8.onnx` 643 MB INT8 | [链接](https://huggingface.co/openai/whisper-large-v3-turbo) | `sherpa-onnx-whisper-turbo` |
| Fun-ASR-Nano-2512（全 int8 + 重复惩罚） | 同上；LLM 为 int8，惩罚 1.2 + 3-gram 屏蔽（fork 旧默认） | 同上 | 984.3M | INT8 100% | `embedding.int8.onnx` 148 MB INT8<br>`encoder_adaptor.int8.onnx` 227 MB INT8<br>`llm.int8.onnx` 573 MB INT8 | [链接](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | `sherpa-onnx-funasr-nano-int8-2025-12-30` |

参数量 = ONNX 文件中所有权重张量的元素总数（含量化 scale 等少量辅助张量）；精度构成按参数量占比统计。INT8 均为 sherpa-onnx 发布包使用的动态量化。

## 5. 测试素材

| 场景 | 素材 | 来源 | 截取区间 | 人工字幕 | 参考字数 |
|---|---|---|---|---|---|
| 会议场景 | 《圆桌派》第八季 EP01：4 人圆桌讨论（优酷官方频道） | [YouTube](https://youtu.be/cguD1ryqDWU) | 10:00 起 8 分钟 | zh-CN，264 条 | 1934 |
| 会议场景 | 《圆桌派》第七季 EP9：李玫瑾/周轶君/马未都，含女声与京腔 | [YouTube](https://youtu.be/iPOMZ3sOwpE) | 15:00 起 8 分钟 | zh-CN，303 条 | 2381 |
| 会议场景 | Drinking Library 播客 EP103（蔡康永）：多人对话，台湾口音 | [YouTube](https://youtu.be/pDxi9MljStw) | 15:00 起 8 分钟 | zh-TW，213 条 | 2281 |
| 会议场景 | Rising Podcast：双人访谈，中英夹杂 | [YouTube](https://youtu.be/EoyaiwVEw4k) | 15:00 起 8 分钟 | zh-Hans，128 条 | 2363 |
| 会议场景 | 一刻talks（蔡康永）：演讲式，有现场观众 | [YouTube](https://youtu.be/K2vBHUUqWoI) | 2:00 起 8 分钟 | zh-Hant，182 条 | 2093 |
| 影视场景 | 《淬火年代》EP01：现代剧，有配乐和音效（正午阳光官方频道） | [YouTube](https://youtu.be/_WYK016EUoc) | 5:00 起 8 分钟 | zh-Hans，116 条 | 682 |
| 影视场景 | 《逐玉》EP01：古装剧，半文言台词（爱奇艺官方） | [YouTube](https://youtu.be/Z0gEjddn8p4) | 10:00 起 8 分钟 | zh-CN，149 条 | 967 |
| 影视场景 | 《三体》EP01：科幻剧，音效重（腾讯视频官方） | [YouTube](https://youtu.be/3-UO8jbrIoM) | 10:00 起 8 分钟 | zh，199 条 | 1354 |
| 影视场景 | 《怒火十二小时》：动作片，枪战与警笛（优酷电影官方）。从 40 分钟起截取：20 分钟处的片段含外语对白和中文翻译字幕，不能作参考 | [YouTube](https://youtu.be/q27grfxz1UA) | 40:00 起 8 分钟 | zh-CN，45 条 | 256 |
| 影视场景 | 《囧爸喜事》：喜剧电影（爱奇艺电影官方） | [YouTube](https://youtu.be/_Np95HVeooE) | 15:00 起 8 分钟 | zh，175 条 | 1280 |

均来自官方频道，只使用上传者提供的人工字幕（不使用 YouTube 自动字幕）。

## 6. 测试环境与方法

- **硬件**：Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz，仅 CPU 推理，6 线程（评测时记录）
- **软件**：sherpa-onnx fork（基于上游 v2.0.1，Python 绑定从源码编译；报告生成时的代码版本 `5c2321f1`），onnxruntime 1.28.2，Python 3.12.11，Windows 11
- **流程**：silero-vad（阈值 0.2，最短静音 0.25 s，最长片段 5 s）切分；VAD 结果按参数和音频内容缓存，**所有模型识别完全相同的片段**；解码批大小：8 个片段/批（SenseVoice-Small（粤语微调版）为整段一次解码）
- **评测规模**：2 个场景 × 5 段 × 8 分钟 = 80 分钟音频，10 个模型配置

## 7. 指标说明

- **CER**（字错误率）=（替换 + 删除 + 插入）/ 参考字数。汉字按字、英文按词计（严格说是 MER）。计算前双方统一做：繁转简、FunASR `cn_tn` 规整（数字转汉字、全角转半角）、去标点、去语气词（嗯/呃/啊/唔）、去 `<sil>` 等特殊标记。场景 CER 按总错误字数 / 总参考字数汇总，不是简单平均
- **替换 / 删除 / 插入**：CER 的三个组成部分（各自占参考字数的比例）。删除高说明漏字，插入高说明多字（复读、幻觉）
- **RTF**（实时率）= 识别耗时 / 音频时长，不含模型加载。RTF 0.1 表示 1 小时音频需 6 分钟
- **加载时间**：创建识别器（加载模型）的耗时
- **幻觉**：参考字幕无人说话处（前后各放宽 0.5 s）模型输出的字数 / 分钟

## 8. 评测中发现并修复的问题

| 提交 | 问题 | 影响 |
|---|---|---|
| `be8cc495` | FireRedASR2-CTC 把 `<sil>` 输出到识别文本 | 字幕中出现 `<sil>`；上游 bug |
| `869f35ce` | Fun-ASR-Nano 默认开启重复惩罚 1.2 + 3-gram 屏蔽 | 掩盖了 int8 LLM 的问题，且让 fp16 LLM 的 CER 从 24.9% 升到 38.9%；现改为默认关闭 |
| `8cdb8be0` | Fun-ASR-Nano 可能复读到 `max_new_tokens` 上限 | 按音频时长限制生成长度后，一段素材 CER 从 33.6% 降到 15.4% |
| `f3460762` | Python 默认 prompt 用半角冒号 `语音转写:` | 丢字、乱码 |
| `99bb5ef6` | 合并上游后 Python TTS 配置绑定编译失败 | Python 包无法构建 |

**Fun-ASR-Nano 的根因**：sherpa-onnx 发布包 `sherpa-onnx-funasr-nano-int8-2025-12-30` 中的 **int8 LLM 量化损坏**。只把 LLM 换成 fp16 后，同一段 101 秒会议音频的 CER 从 134.8% 降到 24.9%。

## 9. 局限与注意事项

- 参考是人工字幕而非逐字稿：会省略口吃重复（"就就就"）、语气词和部分抢话，所以 CER 绝对值偏高；但所有模型面对同一份参考，**排名有效**
- 动作片片段对白很少（45 条字幕），单段 CER 波动较大
- 未测试通用普通话 Paraformer（本次的 Paraformer 是川渝方言微调版），也未测试 GPU 推理和流式模型
- 80 分钟音频属于筛选性评测，正式部署前建议用自己的素材复核

## 10. 复现

```bash
cd scripts/asr-eval
python fetch_material.py && python fetch_material.py --verify
python eval_asr.py            # 断点续跑；--force 重跑，--rescore 只重新打分
python make_report.py         # 生成本报告
```
环境搭建见 [README.md](README.md)。
