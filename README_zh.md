# llm4guandan

**Qwen2.5-14B + LoRA，微调用于[掼蛋](https://zh.wikipedia.org/wiki/%E6%8E%BC%E8%9B%8B)的不完全信息决策。**

本仓库的 LoRA 权重 (`weights/checkpoint-9250/`, 66 MB) 加载在 `Qwen/Qwen2.5-14B-Instruct`
之上，训练集为 46 k 条从 [Danzero+](https://github.com/submit-paper/Danzero_plus)
自博弈环境中生成的掼蛋对局轨迹。训练框架为 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
(SFT、LoRA r=8 α=16 覆盖全部 7 个投影层、DeepSpeed ZeRO-3、bf16、1 epoch)。

推理时通过 `llamafactory-cli api` 启动 vLLM 后端，暴露 OpenAI 兼容接口，然后接入
[LLM4CardGame](https://github.com/THUDM/LLM4CardGame) 的掼蛋 client
(`util/guandan_util/`)，对 [AI4Card](https://github.com/AltmanD/guandan_mcc)
规则 bot 和 Danzero+ 强化学习 bot 做完整对局评测。

---

## 评测结果 (checkpoint-9250, 每对手 100 手)

| 对手 | 手数 | 胜率（正 / 负 reward） | 总 reward | 平均 reward / 手 | LLM 出牌中位耗时 | P99 耗时 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| AI4Card 规则 bot | 102 | **65.7 %** (67 / 35) | **+98** | **+0.961** | 0 秒 | 7 秒 |
| Danzero+ RL bot | 102 | 51.0 % (52 / 50) | +7 | +0.069 | 0 秒 | 9 秒 |

Reward 为 LLM 视角每手的 \{-3, -2, -1, +1, +2, +3\}。延迟为 4×RTX 4090
tensor-parallel-size=4 下相邻两次 LLM 决策的墙钟时间差。

英文 README 见 [README.md](README.md)，完整分布见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 快速上手

```bash
git clone https://github.com/brandonrhan/llm4guandan.git
cd llm4guandan
bash scripts/setup.sh
bash scripts/serve.sh          # 启动 OpenAI 兼容 API :8552
python examples/infer_openai_client.py
```

或者不用 vLLM，直接用 transformers + peft：

```bash
pip install -r requirements.txt
python examples/infer_transformers.py
```

> **⚠️ Prompt 格式非常关键。** 模型是在一份固定的 133 行模板上做的 SFT，
> 输入需要按顺序填 13 个游戏状态槽位，输出是一个 `{"action": [Type, Rank, [Cards]]}`
> 的单行 JSON。写自己的 client 前请先看
> **[`docs/PROMPT_FORMAT.md`](docs/PROMPT_FORMAT.md)** —— 训练时的模板原文见
> [`prompt/prompt_guandan4.py`](prompt/prompt_guandan4.py)，一条真实训练样本见
> [`prompt/sample_training_example.txt`](prompt/sample_training_example.txt)。
> `examples/infer_openai_client.py` 就是照着这个格式写的正确样例。

## 复现训练与评测

```bash
bash scripts/train.sh                                            # 训练
bash scripts/eval.sh weights/checkpoint-9250 100 ai4 my_eval     # 对 AI4 评测 100 手
bash scripts/eval.sh weights/checkpoint-9250 100 danzero my_eval # 对 Danzero+
python scripts/parse_results.py eval_logs/my_eval                # 统计
```

详细步骤见 [`docs/TRAINING.md`](docs/TRAINING.md) 与 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

## License

Apache 2.0，与 Qwen 2.5 base、LLaMA-Factory 保持一致。
