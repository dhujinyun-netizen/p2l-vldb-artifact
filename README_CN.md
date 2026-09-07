# P2L 代码与最小复现包说明

本项目包含 **P2L** 的代码与轻量证据。由于论文尚未正式投稿，公开仓库不包含
主文、补充材料、图、PDF 或投稿专用 LaTeX 文件。公开包不包含模型权重、
M-BEIR 数据、预提取特征、候选语义 ID 缓存或检索输出；这些大文件需按
下述逻辑路径另行准备。

## 1. 方法与最终配置

P2L 面向残差量化 semantic ID 的通用多模态生成式检索。解码器先在
Candidate Trie 中顺序保留一个短前缀，再通过一次神经前向同时得到所有
未决位置的分数，并只比较保留前缀下真实存在的完整叶节点。可选的
Prefix-Conditioned Angular Affinity（PCAA）在同一完整候选前沿上加入经码本
标准化的量化器路径分数，不增加神经预测器前向。

论文锁定配置为：

- semantic-ID 长度 `L=9`；
- prefix length `s=3`，其中包含 modality-routing code；
- prefix/final beam `B=K=50`；
- standardized-cosine PCAA，`lambda=20`；
- inference batch size `8`；
- 单卡评测，使用 `CUDA_VISIBLE_DEVICES=0`。

代码中保留的 `tcis`、`rrg` 和 `rqc` 是历史兼容字段，不代表论文中的额外
方法。论文术语统一为 P2L 与 PCAA。

## 2. 关键目录

```text
src/                         模型、数据接口和检索实现
configs/structnar/           P2L 训练、最终评测和受控配置
configs/genius/              GENIUS 复现配置
scripts/structnar/           评测、统计、绘图和证据审计
scripts/genius/              GENIUS 复现入口
scripts/shared/              M-BEIR 布局与共享数据检查
docs/results/                论文使用的轻量结果、配置哈希和证据记录
```

## 3. 环境与外部资产

推荐使用项目现有 `genius2` 环境。论文记录的主要环境为 Python 3.10、
PyTorch 2.7、CUDA 12.8、Transformers、FAISS、CLIP 和 OmegaConf。硬件或
软件版本变化可能改变绝对耗时，因此速度比较必须固定 checkpoint、候选
集合、batch size、beam、精度模式和计时边界。

完整评测需要：

1. M-BEIR 数据集；
2. P2L checkpoint；
3. GENIUS checkpoint（仅复现系统基线时需要）；
4. 冻结 RQ50 quantizer、候选 semantic IDs、候选特征和 Candidate Trie。

默认逻辑位置为：

```text
mbeir_data/                       M-BEIR 根目录或符号链接
checkpoint/code_tied/             P2L checkpoint
checkpoint/genius/                GENIUS checkpoint
checkpoint/rq_clip_large/         RQ50 quantizer
gen_code/.../cand_pool/           候选 semantic-ID 与 Trie 缓存
```

路径可通过 YAML、`MBEIR_DATA_DIR`、`CKPT_DIR` 和 `CKPT_NAME` 覆盖。

## 4. 最终 CIRR-7 评测

在项目根目录运行：

```bash
MBEIR_DATA_DIR=/path/to/M-BEIR \
CUDA_VISIBLE_DEVICES=0 \
CONFIG_PATH=configs/structnar/final_p2l_d3_rqc_cosine_w20.yaml \
CHECK_EXTRACTED=0 NPROC=1 MASTER_PORT=3971 \
bash scripts/structnar/run_eval.py.sh
```

该配置会重新生成 CIRR-7 LOCAL 的 P2L+PCAA 结果，不默认复用旧 beam。
其他任务和 UNION 评测应从同一配置复制，只修改明确列出的数据集、候选池
和输出路径字段；不得修改 `s`、`B/K`、PCAA 或 checkpoint 后仍称为最终
配置。

## 5. 无权重证据核验

以下命令只读取包内轻量 CSV/JSON/YAML 与论文源文件，不需要数据集或
checkpoint：

```bash
python scripts/structnar/audit_release_evidence.py
```

该便携审计从包内表格重新计算投稿最重要的四组数字：16 个 LOCAL
任务上的 matched `lambda=0` 宏平均及 16/16 改善、CIRR-7 五级候选池的
generation 与 warm decode-to-rerank 降幅、复现 GENIUS-R 与 P2L-R 的
UNION 宏平均及 16/16 改善，以及 5.61M exact GPU FlatIP 参考点。它还
核对这些 headline 是否出现在论文源文件中，并验证发布包的 SHA-256
清单；若纳入 optimized GPU-flat Trie 基线，也会检查其汇总审计结果。

`audit_vldb_evidence.py` 是更深的项目内审计，还会核对原始日志、候选
产物和 checkpoint 哈希；准备好这些外部资产后可在完整工作区运行，但它
不是无权重发布包的默认入口。

## 6. 当前代码能够支持的复现范围

- 在相同 checkpoint、realized semantic-ID index、Trie、beam 与 reranker
  下，`lambda=0` 的 P2L 在 16/16 个 LOCAL 任务上提高 R@10，宏平均从
  39.36 提高到 42.83。
- 在五个嵌套 CIRR-7 候选池上，最终 P2L 配置相对 Sequential+PCAA 将
  decoder-side generation time 降低 51.5%--53.8%，并将 warm
  decode-to-rerank latency 降低 50.5%--53.0%。
- 在完整系统比较中，P2L-R 的 UNION primary-metric macro 从复现的
  GENIUS-R 的 33.92 提高到 39.20，16/16 个任务提高；该比较不能单独归因
  于 P2L，因为两个完整系统的 predictor 与 separately materialized index
  不同。
- Exact GPU FlatIP 在冻结检索空间中更快且 R@10 更高，但占用更多 GPU
  向量存储；它是不同 query semantics 的外部 operating reference，而非
  matched decoder baseline。
- 在 4,170-query CIRR-7 LOCAL manifest 上，用 level-wise CSR 与 batched
  GPU gather 替换顺序基线的 host-side Trie traversal 后，top-50 ID、分数和
  查询输入与原实现逐项完全一致；该优化仅将 59.90 降至 59.33 ms/query，
  而 P2L+PCAA 为 27.65 ms/query，仍快 53.4%。因此主要提速不能归因于一个
  容易修复的 Python Trie 遍历瓶颈。

P2L 是改变 suffix scoring semantics 的 approximate learned-index search
policy，不是保持 Sequential 排序函数不变的 physical-plan rewrite。所有
输出均为 Candidate Trie 中的合法 ID；精确 top-K 保证仅成立于保留前缀下
被完整枚举的候选前沿。
