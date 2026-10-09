# 怎么用

只需要 Python 3.7 以上，不用装任何第三方包。所有命令都在 `track_2a` 目录下运行。

```powershell
cd D:\apertus\uzh-semantic-diff\track_2a
```

| 命令 | 做什么 | 要不要联网 |
|---|---|---|
| `python run.py selftest` | 离线自检（95 项测试），确认代码在你的机器上能跑 | 不要 |
| `python run.py check` | 测接口：能用哪些模型、最短请求、system 角色、强制 JSON、随机种子、生成速度 | 要 |
| `python run.py probe` | 测接口的 7 项扩展能力（prompt_logprobs、预填、n=3、structured_outputs、json_schema、logprobs、前缀缓存），按效果判通过，并给出 D3 / D4 能不能做的结论 | 要 |
| `python run.py smoke` | 12 篇开发文档从头跑到出分，默认跑 m2、d4 和 b0s；`--methods m1,b0,b0s,m2,d4` 可选 | 要 |
| `python run.py run --method m2 --manifest dev60` | 正式批量跑（`--manifest dev60` = 20 页 × 3 语种；不给 manifest 时用 `--split dev/train --limit 20` 取前 N 篇）。`--model 70b` 换成 70B，`--workers 4` 同时处理 4 篇，`--blocks-per-call 3` 每次问几块（0 = 一侧一次问完），`--max-ref-words 1500` 参照文档超过多少词就只给对应位置附近一段 | 要 |
| `python run.py analyze --run runs\<m2目录> --d4-run runs\<d4目录>` | 输出层对比：m2 的两档各自怎么用、d4 的意外程度怎么平滑、两者怎么组合，每种规则一行分数；最后附 m2 三种块判定里金标差异词的比例（要金标，只在开发集上用） | 不要 |
| `python run.py fuse --m2-run runs\<m2目录> --d4-run runs\<d4目录>` | **出最终预测**：把 m2 和 d4 的结果按冻结的规则合成（默认 `--rule verified+d4`），生成官方脚本能读的预测文件并打分；`--rule absent|verified|d4` 可选其他规则。两个输入运行必须覆盖全部文档，缺了会拒绝（`--allow-missing` 才按回退处理） | 不要 |
| `python run.py assemble --runs runs\<fuse-dev目录> runs\<fuse-test目录>` | 把 dev 和 test 两次 fuse 的结果拼成**全集顺序**的提交文件（每语种 224 篇，`data/predictions/full/`），打印两个子集的分数和哈希 | 不要 |
| `python run.py audit --run runs\<fuse目录>` | **验收**：文档是否齐全唯一、m2 每块是否拿到判断（没拿到的按'无差异'记了多少）、d4 每侧数组是否齐全且长度等于词数、对齐是否 exact、退回 m2 的侧数、预测文件行数/顺序/哈希。只检查不改 | 不要 |
| `python run.py stats --run runs\<fuse目录> --vs ctfalign|d4|verified|absent` | 配对、按英文页面分组的 bootstrap（三语种同页同抽，重算官方口径），给原始点估计、重采样均值、95% 区间和差值不为正的次数 | 不要 |
| `python run.py compare --run runs\<目录>` | 这次运行里的方法和两个参照系统（CTFAlign、mmBERT-SimCSE）在同一批文档上的对比表 | 不要 |
| `python run.py score --pred-dir <目录> --system <名字> --split dev` | 给已有的预测文件算分 | 不要 |
| `python run.py viewer --allow-test` | **离线查看器**：把 `data/predictions/full/` 的预测和金标做成一个单文件 HTML（`demo/viewer.html`，约 8 MB，不联网）：两篇文档并排、按分数着色、金标画下划线、CTFAlign 可切换对照、逐篇和分组的 Spearman。`--pred-dir`/`--system`/`--out` 可指向别的预测文件；不传 `--allow-test` 只放 dev 文档 | 不要 |

第一次联网运行会问一次 API key，之后存在 `track_2a/.env` 里，不再问。

## 方法

- **b0**：论文原版的基线提示词。让模型给两篇文档的每个词打 0–5 分。题目问的就是“能不能超过它”。
- **m1**：把两篇文档切成带编号的小块一起给模型，让它列出差异，每条差异要原样引用原文。程序再把引用定位回原文的词，被引用到的词记 1，其余记 0。第一次实跑证明它抄得太长、长文只看开头，分数接近 0。
- **b0s**：同一个论文提示词，但按论文自己的 250 词切块方式跑（每块一次调用），这样不会被 60 秒超时打断。报告里的"基线提示词做法"用它。
- **m2**：逐块提问。每次给模型另一篇文档做参照（超过 1,500 词时只给对应位置附近的 1,500 词），再给这一篇的 3 个小块，让它对每块判定：全部有对应（covered）、整块没有对应（absent）、部分有差异（partly，并原样引用有差异的几个词）。两个方向都问一遍。每个词记下来自哪一档（整块无对应 / 引用），成绩表里给出两档各自的精度。dev60 实测：absent 块里金标差异率 de 0.53 / fr 0.27 / it 0.40（有信息），partly 块里和整体一样（没信息），引用词精度 ≈ 金标差异率（没信息）——所以最终只用 absent 这一档，再交给 d4 核实。

- **d4**：不让模型写逐词标签或解释（每次请求最多解码 1 个 token），让它"照着念"。把另一篇文档当翻译原文放在 user 里，把要打分的这一篇预填在 assistant 里，只生成 1 个 token，但要回每个 prompt token 的概率。看过对面之后仍然很意外的词，就是对面解释不了的词。再不给对面跑一次，得到"对面帮了多少"。每篇文档 4 次调用、约 2.6 秒，输出是连续分数。

- **fuse（输出层，最终预测用它）**：m2 和 d4 各自的毛病互补。m2 说"整块没对应"的块里有一部分是看走眼（法语尤其多），d4 说某个词意外却说不清边界。合成分两步：① **核实**——m2 判为"整块没对应"的块，只有块里词的 d4 信号（看过对面后的意外程度 s_c 减去 0.5 倍不看对面的意外程度 s_u）的均值达到本篇 70% 分位才留下，剩下的是 m2 判错的；② **排序**——比赛指标是把一个语种所有词放在一起算 Spearman，留白的 90% 词互相打平是浪费，所以核实过的块排最前，其余所有词按 d4 信号（s_c − 0.75·s_u，±4 词滑动平均）排序。排序分数经过 `squash`：负值（看过对面之后比不看还不意外，即"对面解释得很好"的词，占 62–70%）一律截成 0 并列在最底，正值单调压到 (0, 1)。这个"解释底板"是方法的一部分，不是无损缩放：dev 上截断比保留负值高约 0.02。所有参数在 dev60 上挑、在 dev/val 上确认，不碰金标；参数定下之后在 dev/val 上做过一次敏感性扫描（只看、没改）。dev60 上：m2 只用整块无对应 0.266 → 核实 0.358 → 核实+排序 0.375（CTFAlign 同批 0.333）。

接口每次请求最多约 60 秒，超过就返回 504。所以任何方法都要控制单次输出长度。接口支持 prompt_logprobs、assistant 预填、n=3、structured_outputs、json_schema、前缀缓存（10 月 6 日 `probe` 实测，vLLM 构建 `0.23.1rc1.dev1029+ga601a9d99`）。

冻结运行的逐篇结果（m2 / d4 / fuse 的 dev 与 test，b0s 的 dev60 与 test）压缩存在 `data/frozen_runs/`，不用调模型就能重算报告里的全部数字：`python tools/report_tables.py`（表 1 和消融）、`python run.py stats --run data/frozen_runs/20261007-122413-fuse-verified_d4-test --vs ctfalign --allow-test`（bootstrap）、`python run.py audit --run data/frozen_runs/<fuse目录> [--allow-test]`（验收）。

基线 b0s 在 test 上的预测、成绩表和失败画像在 `data/baselines/b0s/test/`（诊断脚本 `tools/b0s_chunk_diagnostic.py`，需要 runs/ 里的原始运行目录）。

在 Euler 上跑见 `euler/README.md`。

## Docker / `make run`（评委怎么跑）

`make run` 构建镜像（`python:3.11-slim`，不装任何包）并在容器里执行 `src/entrypoint.sh`：自检 → 测接口 → m2 → d4 → fuse → audit，预测文件和验收摘要复制到 `output/<split>/`，原始请求和缓存留在 `runs/`（挂载到宿主机，中断后重跑会接着跑）。变量：`LLM_API_KEY`（必填）、`LLM_BASE_URL`、`LLM_NAME`、`SPLIT=test|dev|dev/val|smoke`（默认 test，约 25 分钟）、`WORKERS`（默认 2）、`SYSTEM`（预测文件名前缀，默认 tsemin）。`make smoke` = 12 篇 3 分钟；`make selftest` = 离线自检。输入文件只要有 `text_a`/`text_b` 就能出预测，没有金标就不打分。

Windows 没有 make 时直接用 docker 命令：
```powershell
docker build -t tsemin-swissgov-rsd .
$env:LLM_API_KEY = "<your key>"
docker run --rm -e LLM_API_KEY=$env:LLM_API_KEY -e SPLIT=smoke -v ${PWD}\runs:/work/runs -v ${PWD}\output:/work/output tsemin-swissgov-rsd
```

## 结果在哪

每次运行在 `runs/<时间-名字>/` 下生成：

| 文件 | 内容 |
|---|---|
| `summary_<方法>.txt` | 成绩表、失败清单、调用次数和 token 数（`fuse` 的还带参数） |
| `results_<方法>.jsonl` | 每篇一行：状态、每个词的预测、模型引用了什么、每条引用定位到哪 |
| `predictions/<方法>/<名字>_admin_<语种>.jsonl.jsonl` | 官方评分脚本能直接读的预测文件 |
| `calls.jsonl` | 每次请求和模型的原始返回 |

`runs/_cache/` 是缓存：同样的请求不会发第二次，中途断了重跑同一条命令会接着跑。

## 成绩表怎么看

- **Spearman**：比赛的主指标。三个语种的平均值是最终比的数。
- **准确率**：模型标成“有差异”的词里，金标也是差异的比例。
- **召回率**：金标是差异的词里，模型标出来的比例。
- **块判定**（m2）：`covered` / `absent` / `partly` 各有多少块。
- **引用定位**：`ok` 找到了；`ambiguous` 同一块里出现多次，取第一处；`elsewhere` 不在模型说的那块、但全文只有一处；`not_found` 找不到，这条引用作废。
- **按文档位置十等分的召回**：把每篇文档从头到尾分成十段，看每段里金标的差异词被找出来多少。后面几段明显低，就是"只读开头"。
- **分档精度**：来自"整块无对应"的词和来自引用的词，各自有多大比例真是差异。哪档精度高，哪档就该排前面；精度接近金标差异率（10–18%）的档等于白标。

12 篇的 Spearman 只用来确认流程通了，篇数太少，不能当成绩看（12 篇上 m2 只用整块无对应是 0.453，到 60 篇掉到 0.266）。

官方指标是把一个语种所有文档的词放在一起算一个 Spearman（不是每篇算完取平均），再对三个语种平均。所以①全篇预测 0 的文档不会被跳过，它的词都在池子里打平；②分数跨文档可比很重要，连续分不要按每篇归一化。

## 规矩

- test 集不在这个目录里，代码也默认拒绝读它。
- 示例、调参只用 dev/train；dev/val 留着做确认。
