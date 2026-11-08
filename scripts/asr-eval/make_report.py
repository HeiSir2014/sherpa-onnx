#!/usr/bin/env python3
"""Generate REPORT.md (Chinese) from model files and evaluation results.

Parameter counts, precision and sizes are read from the ONNX files; metrics
come from work/results. Re-run after any evaluation to refresh the report:

  python make_report.py
"""
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

import onnx

from eval_asr import REPO_ROOT, VAD_PARAMS, WORK, _model_dir
from fetch_material import clip_paths
from metrics import error_breakdown, parse_srt

HERE = Path(__file__).resolve().parent
FUNASR_INT8 = "sherpa-onnx-funasr-nano-int8-2025-12-30"

# Display metadata. Files are (model dir, glob) pairs; dirs are looked up in
# work/models first, then in the repo root (same rule as eval_asr.py).
MODEL_INFO = {
    "fire-red-aed": dict(
        name="FireRedASR2-AED",
        arch="Conformer 编码器 + Transformer 注意力解码器（自回归）",
        source="https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED",
        package="sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26",
        langs="普通话（含 20+ 方言/口音）、英语",
        files=[("sherpa-onnx-fire-red-asr2-zh_en-int8-2026-02-26",
                "*.onnx")]),
    "qwen3-asr-0.6b": dict(
        name="Qwen3-ASR-0.6B",
        arch="音频编码器 + Qwen3 LLM 解码器（自回归）",
        source="https://github.com/QwenLM/Qwen3-ASR",
        package="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        langs="30 种语言 + 22 种方言",
        files=[("sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25", "*.onnx")]),
    "fire-red-ctc": dict(
        name="FireRedASR2-CTC",
        arch="FireRedASR2-AED 的 Conformer 编码器 + CTC 分支（非自回归）",
        source="https://www.modelscope.cn/models/FireRedTeam/FireRedASR2-AED",
        package="sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25",
        langs="普通话、英语",
        files=[("sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25",
                "*.onnx")]),
    "sv-funasr-nano-ctc": dict(
        name="SenseVoice-FunASR-Nano-CTC",
        arch="Fun-ASR-Nano-2512 的音频编码器 + CTC 头（无 LLM，非自回归）",
        source="https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
        package="sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17",
        langs="中文（7 大方言、26 地区口音）、英语、日语等",
        files=[("sherpa-onnx-sense-voice-funasr-nano-int8-2025-12-17",
                "*.onnx")]),
    "sense-voice-2024": dict(
        name="SenseVoice-Small（原版）",
        arch="SAN-M 编码器 + CTC（非自回归）",
        source="https://www.modelscope.cn/models/iic/SenseVoiceSmall",
        package="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
        langs="普通话、粤语、英语、日语、韩语",
        files=[("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
                "*.onnx")]),
    "sense-voice": dict(
        name="SenseVoice-Small（粤语微调版）",
        arch="SAN-M 编码器 + CTC（非自回归）；原版再用 2.18 万小时粤语微调",
        source="https://huggingface.co/ASLP-lab/WSYue-ASR",
        package="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
        langs="粤语为主，兼顾普通话、英语、日语、韩语",
        files=[("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09",
                "*.onnx")]),
    "funasr-nano-fp16": dict(
        name="Fun-ASR-Nano-2512（LLM fp16）",
        arch="音频编码器 + 适配器 + Qwen3-0.6B LLM（自回归）",
        source="https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
        package=f"{FUNASR_INT8} 的编码器/embedding + ModelScope "
                "zengshuishui/FunASR-nano-onnx 的 llm_fp16",
        langs="中文（7 大方言、26 地区口音）、英语、日语等 31 种",
        files=[(FUNASR_INT8, "encoder_adaptor.int8.onnx"),
               (FUNASR_INT8, "embedding.int8.onnx"),
               ("funasr-nano-llm-fp16", "llm.fp16.onnx")]),
    "paraformer": dict(
        name="Paraformer-large（川渝方言微调版）",
        arch="Paraformer 非自回归编解码",
        source="https://huggingface.co/ASLP-lab/WSChuan-ASR",
        package="sherpa-onnx-paraformer-zh-int8-2025-10-07",
        langs="普通话、川渝方言",
        files=[("sherpa-onnx-paraformer-zh-int8-2025-10-07", "*.onnx")]),
    "whisper-turbo": dict(
        name="Whisper large-v3-turbo",
        arch="Transformer 编解码（自回归）",
        source="https://huggingface.co/openai/whisper-large-v3-turbo",
        package="sherpa-onnx-whisper-turbo",
        langs="99 种语言",
        files=[("sherpa-onnx-whisper-turbo", "*.int8.onnx")]),
    "funasr-nano-int8-pen": dict(
        name="Fun-ASR-Nano-2512（全 int8 + 重复惩罚）",
        arch="同上；LLM 为 int8，惩罚 1.2 + 3-gram 屏蔽（fork 旧默认）",
        source="https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
        package=FUNASR_INT8,
        langs="同上",
        files=[(FUNASR_INT8, "*.onnx")]),
}

# RTF measured while other CPU-heavy work (builds/diagnostics) was running
RTF_CAVEATS = {
    "whisper-turbo": "会议素材中有 4 段运行时机器在编译，RTF 偏高",
    "paraformer": "部分素材运行时有诊断任务并行，RTF 可能略高",
}

_BITS = {onnx.TensorProto.INT8: "INT8", onnx.TensorProto.UINT8: "INT8",
         onnx.TensorProto.FLOAT16: "FP16", onnx.TensorProto.FLOAT: "FP32"}


def onnx_stats(path: Path):
    """(parameter count, Counter of params by precision) of one ONNX file."""
    model = onnx.load(str(path), load_external_data=False)
    by_type = Counter()
    for init in model.graph.initializer:
        numel = 1
        for d in init.dims:
            numel *= d
        by_type[_BITS.get(init.data_type, "其他")] += numel
    return sum(by_type.values()), by_type


def model_inventory(key):
    files = []
    for d, pattern in MODEL_INFO[key]["files"]:
        files += sorted(_model_dir(d).glob(pattern))
    total_params, total_bytes, by_type, details = 0, 0, Counter(), []
    for f in files:
        n, t = onnx_stats(f)
        size = f.stat().st_size
        total_params += n
        total_bytes += size
        by_type += t
        main = max(t, key=t.get) if t else "-"
        details.append(f"`{f.name}` {size / 2**20:.0f} MB {main}")
    float_params = sum(v for k, v in by_type.items() if k != "其他")
    mix = " / ".join(f"{k} {v / float_params:.0%}"
                     for k, v in by_type.most_common() if k != "其他"
                     and v / float_params >= 0.01)
    return dict(params=total_params, bytes=total_bytes, mix=mix,
                details="<br>".join(details))


def load_results():
    rows = {}
    for p in (WORK / "results").glob("*/*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        clip = {"scenario": r["scenario"], "name": r["clip"]}
        ref = parse_srt(clip_paths(clip)[1].read_text(encoding="utf-8"))
        hyp = parse_srt(p.with_suffix(".srt").read_text(encoding="utf-8"))
        r.update(error_breakdown(" ".join(c.text for c in ref),
                                 " ".join(c.text for c in hyp)))
        rows[(r["model"], r["clip"])] = r
    return rows


def aggregate(rows, model, clips):
    rs = [rows[(model, c["name"])] for c in clips]
    n = sum(r["ref_tokens"] for r in rs)
    dur = sum(r["duration"] for r in rs)
    s, d, i = (sum(r[k] for r in rs) for k in ("sub", "del", "ins"))
    return dict(
        cer=(s + d + i) / n, sub=s / n, dele=d / n, ins=i / n,
        rtf=sum(r["rtf"] * r["duration"] for r in rs) / dur,
        halluc=sum(r["halluc_per_min"] * r["duration"] for r in rs) / dur,
        load=max(r["load_seconds"] for r in rs))


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                   text=True).strip()


def scenario_stats(rows, clips, scen):
    cs = [c for c in clips if c["scenario"] == scen]
    models = [m for m in MODEL_INFO
              if all((m, c["name"]) in rows for c in cs)]
    agg = {m: aggregate(rows, m, cs) for m in models}
    order = sorted(models, key=lambda m: agg[m]["cer"])
    best = {c["name"]: min(rows[(m, c["name"])]["cer"] for m in models)
            for c in cs}
    wins = Counter(m for c in cs for m in models
                   if rows[(m, c["name"])]["cer"] == best[c["name"]])
    return dict(clips=cs, agg=agg, order=order, best=best, wins=wins)


def render_recommendations(w, stats):
    meet, film = stats["meeting"], stats["film"]
    name = {k: v["name"] for k, v in MODEL_INFO.items()}

    def cer(s, m):
        return s["agg"][m]["cer"]

    def rtf(s, m):
        return s["agg"][m]["rtf"]

    top_m, top_f = meet["order"][0], film["order"][0]
    fast = "sv-funasr-nano-ctc"
    w("## 1. 结论与推荐\n")
    w("> 推荐结论为人工判断；理由中的数字由评测结果自动计算。\n")
    w("| 用途 | 推荐模型 | 理由 |\n|---|---|---|")
    w(f"| 会议实时字幕 | **{name[fast]}** + VAD | 会议 CER {cer(meet, fast):.2%}，"
      f"比第一名（{name[top_m]}）只差 {(cer(meet, fast) - cer(meet, top_m)) * 100:.1f} "
      f"个百分点；RTF {rtf(meet, fast):.3f}，速度是它的 "
      f"{rtf(meet, top_m) / rtf(meet, fast):.0f} 倍 |")
    w(f"| 会后纪要 | **{name[top_m]}**（要热词时用 {name['qwen3-asr-0.6b']}） | "
      f"会议总 CER 第一（{cer(meet, top_m):.2%}），{len(meet['clips'])} 段中 "
      f"{meet['wins'][top_m]} 段最佳；sherpa-onnx 里的 FireRed 不支持热词 |")
    w(f"| 影视字幕（质量优先） | **{name[top_f]}** | 影视总 CER 第一"
      f"（{cer(film, top_f):.2%}），{len(film['clips'])} 段中 "
      f"{film['wins'][top_f]} 段最佳 |")
    w(f"| 影视字幕（成本优先） | **{name['sense-voice-2024']}** | 影视 CER "
      f"{cer(film, 'sense-voice-2024'):.2%}，RTF {rtf(film, 'sense-voice-2024'):.3f}；"
      "CTC 自带真实 token 时间戳，便于对齐字幕时间轴 |")
    w("| 不推荐 | Fun-ASR-Nano（LLM 版）、Whisper large-v3-turbo；"
      "普通话场景不用两个方言微调版（SenseVoice 粤语版、Paraformer 川渝版） | "
      f"Whisper 会议/影视 CER {cer(meet, 'whisper-turbo'):.1%}/"
      f"{cer(film, 'whisper-turbo'):.1%}；Fun-ASR-Nano（fp16）影视 CER "
      f"{cer(film, 'funasr-nano-fp16'):.1%}，会议 RTF "
      f"{rtf(meet, 'funasr-nano-fp16'):.2f}；方言微调版不是通用普通话模型 |\n")


def render(rows, inventory, clips):
    scen_names = {"meeting": "会议场景", "film": "影视场景"}
    out = []
    w = out.append
    w("# 本地 ASR 模型评测报告：会议 vs 影视字幕\n")
    w("> 本文件由 `scripts/asr-eval/make_report.py` 根据评测结果自动生成，"
      "请勿手工修改；重新评测后运行 `python make_report.py` 即可更新。\n")

    stats = {s: scenario_stats(rows, clips, s) for s in ("meeting", "film")}
    render_recommendations(w, stats)

    for scen in ("meeting", "film"):
        cs, agg, order = (stats[scen][k] for k in ("clips", "agg", "order"))
        idx = 2 if scen == "meeting" else 3
        w(f"## {idx}. {scen_names[scen]}总体结果（{len(cs)} 段 × 8 分钟）\n")
        w("| 排名 | 模型 | 精度 | 参数量 | 模型大小 | CER | 替换 | 删除 | 插入 | "
          "RTF | 1 小时音频耗时 | 加载时间 | 幻觉（字/分钟） |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for rank, m in enumerate(order, 1):
            a, inv = agg[m], inventory[m]
            note = " ⚠" if m in RTF_CAVEATS else ""
            w(f"| {rank} | {MODEL_INFO[m]['name']} | {inv['mix']} | "
              f"{inv['params'] / 1e6:.0f}M | {inv['bytes'] / 2**20:.0f} MB | "
              f"**{a['cer']:.2%}** | {a['sub']:.2%} | {a['dele']:.2%} | "
              f"{a['ins']:.2%} | {a['rtf']:.3f}{note} | "
              f"{a['rtf'] * 60:.1f} 分钟 | {a['load']:.1f} s | "
              f"{a['halluc']:.1f} |")
        w("")
        w("逐段 CER：\n")
        w("| 模型 | " + " | ".join(c["name"].split("-", 1)[1] for c in cs) +
          " |")
        w("|---|" + "---|" * len(cs))
        best = stats[scen]["best"]
        for m in order:
            cells = []
            for c in cs:
                v = rows[(m, c["name"])]["cer"]
                cells.append(f"**{v:.1%}**" if v == best[c["name"]]
                             else f"{v:.1%}")
            w(f"| {MODEL_INFO[m]['name']} | " + " | ".join(cells) + " |")
        w("\n（加粗为该段最佳）\n")
    caveats = "；".join(f"{MODEL_INFO[k]['name']}：{v}"
                       for k, v in RTF_CAVEATS.items())
    w(f"⚠ {caveats}。\n")

    w("## 4. 模型清单\n")
    w("| 模型 | 结构 | 语言 | 参数量 | 精度构成 | 文件 | 来源 | sherpa-onnx 模型包 |")
    w("|---|---|---|---|---|---|---|---|")
    for m, info in MODEL_INFO.items():
        inv = inventory[m]
        w(f"| {info['name']} | {info['arch']} | {info['langs']} | "
          f"{inv['params'] / 1e6:.1f}M | {inv['mix']} | {inv['details']} | "
          f"[链接]({info['source']}) | `{info['package']}` |")
    w("\n参数量 = ONNX 文件中所有权重张量的元素总数（含量化 scale 等少量辅助张量）；"
      "精度构成按参数量占比统计。INT8 均为 sherpa-onnx 发布包使用的动态量化。\n")

    w("## 5. 测试素材\n")
    w("| 场景 | 素材 | 来源 | 截取区间 | 人工字幕 | 参考字数 |")
    w("|---|---|---|---|---|---|")
    for c in clips:
        ref = parse_srt(clip_paths(c)[1].read_text(encoding="utf-8"))
        chars = sum(len(x.text.replace(" ", "")) for x in ref)
        start = f"{c['start'] // 60}:{c['start'] % 60:02d}"
        w(f"| {scen_names[c['scenario']]} | {c['note']} | "
          f"[YouTube](https://youtu.be/{c['id']}) | {start} 起 "
          f"{c['duration'] // 60} 分钟 | {c['sub_lang']}，{len(ref)} 条 | "
          f"{chars} |")
    w("\n均来自官方频道，只使用上传者提供的人工字幕（不使用 YouTube 自动字幕）。\n")

    w("## 6. 测试环境与方法\n")
    threads, batches = _run_settings(rows)
    w(f"- **硬件**：{_cpu()}，仅 CPU 推理，{threads} 线程（评测时记录）\n"
      "- **软件**：sherpa-onnx fork（基于上游 v2.0.1，Python 绑定从源码编译；"
      f"报告生成时的代码版本 `{_git('rev-parse', '--short', 'HEAD')}`），"
      f"onnxruntime 1.28.2，Python {platform.python_version()}，"
      f"{platform.system()} {platform.release()}\n"
      f"- **流程**：silero-vad（阈值 {VAD_PARAMS['threshold']}，最短静音 "
      f"{VAD_PARAMS['min_silence_duration']} s，最长片段 "
      f"{VAD_PARAMS['max_speech_duration']} s）切分；VAD 结果按参数和音频内容缓存，"
      f"**所有模型识别完全相同的片段**；解码批大小：{batches}\n"
      "- **评测规模**：2 个场景 × 5 段 × 8 分钟 = 80 分钟音频，"
      f"{len(MODEL_INFO)} 个模型配置\n")

    w("## 7. 指标说明\n")
    w("- **CER**（字错误率）=（替换 + 删除 + 插入）/ 参考字数。汉字按字、英文按词计"
      "（严格说是 MER）。计算前双方统一做：繁转简、FunASR `cn_tn` 规整"
      "（数字转汉字、全角转半角）、去标点、去语气词（嗯/呃/啊/唔）、"
      "去 `<sil>` 等特殊标记。场景 CER 按总错误字数 / 总参考字数汇总，不是简单平均\n"
      "- **替换 / 删除 / 插入**：CER 的三个组成部分（各自占参考字数的比例）。"
      "删除高说明漏字，插入高说明多字（复读、幻觉）\n"
      "- **RTF**（实时率）= 识别耗时 / 音频时长，不含模型加载。"
      "RTF 0.1 表示 1 小时音频需 6 分钟\n"
      "- **加载时间**：创建识别器（加载模型）的耗时\n"
      "- **幻觉**：参考字幕无人说话处（前后各放宽 0.5 s）模型输出的字数 / 分钟\n")

    w("## 8. 评测中发现并修复的问题\n")
    w("| 提交 | 问题 | 影响 |\n|---|---|---|")
    w("| `be8cc495` | FireRedASR2-CTC 把 `<sil>` 输出到识别文本 | 字幕中出现 `<sil>`；"
      "上游 bug |")
    w("| `869f35ce` | Fun-ASR-Nano 默认开启重复惩罚 1.2 + 3-gram 屏蔽 | 掩盖了 int8 LLM "
      "的问题，且让 fp16 LLM 的 CER 从 24.9% 升到 38.9%；现改为默认关闭 |")
    w("| `8cdb8be0` | Fun-ASR-Nano 可能复读到 `max_new_tokens` 上限 | 按音频时长限制"
      "生成长度后，一段素材 CER 从 33.6% 降到 15.4% |")
    w("| `f3460762` | Python 默认 prompt 用半角冒号 `语音转写:` | 丢字、乱码 |")
    w("| `99bb5ef6` | 合并上游后 Python TTS 配置绑定编译失败 | Python 包无法构建 |")
    w("\n**Fun-ASR-Nano 的根因**：sherpa-onnx 发布包 "
      f"`{FUNASR_INT8}` 中的 **int8 LLM 量化损坏**。只把 LLM 换成 fp16 后，"
      "同一段 101 秒会议音频的 CER 从 134.8% 降到 24.9%。\n")

    w("## 9. 局限与注意事项\n")
    w("- 参考是人工字幕而非逐字稿：会省略口吃重复（\"就就就\"）、语气词和部分抢话，"
      "所以 CER 绝对值偏高；但所有模型面对同一份参考，**排名有效**\n"
      "- 动作片片段对白很少（45 条字幕），单段 CER 波动较大\n"
      "- 未测试通用普通话 Paraformer（本次的 Paraformer 是川渝方言微调版），"
      "也未测试 GPU 推理和流式模型\n"
      "- 80 分钟音频属于筛选性评测，正式部署前建议用自己的素材复核\n")

    w("## 10. 复现\n")
    w("```bash\ncd scripts/asr-eval\n"
      "python fetch_material.py && python fetch_material.py --verify\n"
      "python eval_asr.py            # 断点续跑；--force 重跑，--rescore 只重新打分\n"
      "python make_report.py         # 生成本报告\n```\n"
      "环境搭建见 [README.md](README.md)。")
    return "\n".join(out) + "\n"


def _cpu():
    try:
        name = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        name = platform.processor()
    return name


def _run_settings(rows):
    """Thread count and batch sizes as recorded in the results."""
    threads = sorted({str(r.get("num_threads", "未记录")) for r in rows.values()})
    by_batch = {}
    for r in rows.values():
        by_batch.setdefault(str(r.get("batch_size", "未记录")), set()).add(
            MODEL_INFO.get(r["model"], {}).get("name", r["model"]))
    if len(by_batch) == 1:
        batches = f"{next(iter(by_batch))} 个片段/批"
    else:
        main = max(by_batch, key=lambda b: len(by_batch[b]))
        others = "；".join(
            f"{'、'.join(sorted(ms))}为"
            f"{'整段一次解码' if b == 'all' else b + ' 个片段/批'}"
            for b, ms in by_batch.items() if b != main)
        batches = f"{main} 个片段/批（{others}）"
    return "/".join(threads), batches


def main():
    clips = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    rows = load_results()
    inventory = {m: model_inventory(m) for m in MODEL_INFO}
    (HERE / "REPORT.md").write_text(render(rows, inventory, clips),
                                    encoding="utf-8")
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
