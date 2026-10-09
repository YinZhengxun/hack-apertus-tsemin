# Independent re-run of `make run` (default SPLIT=test), 9 Oct 2026

Machine: the team's Windows PC (Docker Desktop 4.94), CSCS endpoint, Apertus-v1.5-8B, two days after the frozen run on Euler.
First attempt (11:54–12:14 CEST): m2 2,077 live requests, 7.5 s per pair; 2 of 672 d4 requests failed with HTTP 500 even after the client's retries
(admin_de_86, admin_de_137), the fuse fell back to m2 for those two sides and the audit stopped the run (exit 3, nothing delivered).
Second attempt (12:33–12:34) after adding the automatic repetition of inference steps: all other requests from the cache, the failed ones re-sent;
168/168 documents complete, audit without fatal findings, predictions delivered. Score of the re-run vs the submitted run: 0.322 vs 0.323
(de .412 / .412, fr .189 / .189, it .365 / .367); the prediction files differ (sha256 da325000…, e34cc7f5…, 5ee19781… vs ea93e63f…, f265b513…, 01182032…).

## summary_fuse.txt

```
输出规则 verified+d4  参数 {"rule": "verified+d4", "quantile": 0.7, "verify_coef": 0.5, "rank_coef": 0.75, "window": 4, "scale": 3.0}
方法 fuse | 模型 swiss-ai/Apertus-v1.5-8B | 提示词 fuse:verified+d4
语种  篇数  计分词数  Spearman   准确率   召回率    F1   金标差异率  预测差异率  失败篇数
de      56     40689     0.412    0.277    0.783  0.409      0.107      0.303        0
fr      56     46623     0.189    0.143    0.640  0.233      0.089      0.397        0
it      56     44768     0.365    0.313    0.743  0.440      0.164      0.390        0
三个语种平均 Spearman: 0.322
文档状态: ok=168
  de: 按文档位置十等分的召回: 0.66 0.83 0.76 0.77 0.79 0.76 0.72 0.83 0.77 0.88
  fr: 按文档位置十等分的召回: 0.65 0.70 0.53 0.53 0.70 0.71 0.66 0.65 0.64 0.61
  it: 按文档位置十等分的召回: 0.70 0.83 0.76 0.71 0.74 0.73 0.67 0.79 0.78 0.72
开销: 168 篇, 2749 次调用 (其中缓存 2748), 每篇进 28519 / 出 1059 token, 每篇 0.6 秒
```

## audit.txt

```
验收摘要（只检查，不改任何预测）
fuse: runs/20261009-123417-fuse-test-20261009123302-1
m2: runs/20261009-123335-m2-test-20261009123302-1
d4: runs/20261009-123403-d4-test-20261009123302-1

[de] 文档 56
  m2 结果缺的文档: 0
  d4 结果缺的文档: 0
  fuse 结果缺的文档: 0
  m2 块: 问了 2035，拿到判断 2035，没拿到 0（按'无差异'记）；判断分布 {'partly': 1103, 'absent': 258, 'covered': 674}；受影响文档 0
  d4 侧: 应有 112，完整 112，缺 0，长度不符 0；对齐 {'exact': 224}
  fuse 状态 {'ok': 56}；没有 d4、退回 m2 的侧 0
  预测文件: 56 行，顺序与金标一致 True，长度不符 0，取值 [0.000, 0.949]，sha256 da325000604c627e
[fr] 文档 56
  m2 结果缺的文档: 0
  d4 结果缺的文档: 0
  fuse 结果缺的文档: 0
  m2 块: 问了 1952，拿到判断 1949，没拿到 3（按'无差异'记）；判断分布 {'partly': 1169, 'covered': 535, 'absent': 245}；受影响文档 2
  d4 侧: 应有 112，完整 112，缺 0，长度不符 0；对齐 {'exact': 224}
  fuse 状态 {'ok': 56}；没有 d4、退回 m2 的侧 0
  预测文件: 56 行，顺序与金标一致 True，长度不符 0，取值 [0.000, 0.920]，sha256 e34cc7f5aab50192
[it] 文档 56
  m2 结果缺的文档: 0
  d4 结果缺的文档: 0
  fuse 结果缺的文档: 0
  m2 块: 问了 1901，拿到判断 1896，没拿到 5（按'无差异'记）；判断分布 {'partly': 1160, 'absent': 237, 'covered': 499}；受影响文档 3
  d4 侧: 应有 112，完整 112，缺 0，长度不符 0；对齐 {'exact': 224}
  fuse 状态 {'ok': 56}；没有 d4、退回 m2 的侧 0
  预测文件: 56 行，顺序与金标一致 True，长度不符 0，取值 [0.000, 0.901]，sha256 5ee19781bcf31c42

已声明的限制（不改预测，报告里写明）:
  - fr: 3 of 1952 m2 blocks were not judged (kept the default 'no difference') in 2 documents, 0 of them recorded as ok
  - it: 5 of 1901 m2 blocks were not judged (kept the default 'no difference') in 3 documents, 0 of them recorded as ok
没有致命问题。
```
