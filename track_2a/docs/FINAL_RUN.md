# 最终一次运行（冻结后才做，只做一次）

前提：方法和 `fuse` 的参数已经在 dev 上定下并写进报告；dev 的所有数字都已经算完。

1. 取 test 金标文件（数据集仓库 `ZurichNLP/SwissGov-RSD`，commit `1807a42100e742ed03d337c54c4b9ea86995f565`，目录 `data/evaluation/gold_labels/test/`）：`gold_admin_de.jsonl`、`gold_admin_fr.jsonl`、`gold_admin_it.jsonl`（每个 56 行；不要用 `_short` 文件），复制到 `track_2a/data/gold/test/`。

2. 跑两个方法（登录节点 tmux；模型按冻结的选择，8B 或 `--model 70b`）：

   ```bash
   cd ~/uzh-semantic-diff/track_2a
   tmux new -d -s m2test 'python3 run.py run --method m2 --split test --allow-test --workers 2 2>&1 | tee runs/tmux_m2test.log'
   tmux new -d -s d4test 'python3 run.py run --method d4 --split test --allow-test --workers 2 2>&1 | tee runs/tmux_d4test.log'
   ```

3. 合成最终预测：

   ```bash
   python3 run.py fuse --m2-run runs/<m2-test目录> --d4-run runs/<d4-test目录> --allow-test --system tsemin
   ```

   预测文件在 `runs/<fuse目录>/predictions/fuse/tsemin_admin_{de,fr,it}.jsonl.jsonl`，五字段、按官方金标顺序，官方脚本直接读：

   ```bash
   python scripts/evaluate_predictions_admin.py --predictions-path-prefix <目录>/tsemin_admin_ --split test
   ```

   路径里不能出现 `llm` 三个字母（官方脚本遇到会走另一条分支并丢掉一批文档）。

4. 任何一次调用失败（summary 里的"未正常完成"），重跑同一条 `run` 命令——缓存会只补失败的那几次——再重新 `fuse`。

5. 把 `runs/<三个目录>` 整个拷回本机归档；报告里写 test 的分数时注明"冻结后单次运行"。

6. **提交用的文件要按全集顺序**（主办方 10-08 答疑：评委在 test 上打分，但预测文件请按 224 篇的 full 顺序提交；官方脚本会自己从里面挑出 test 的 56 篇）：

   ```bash
   python3 run.py assemble --runs runs/<fuse-dev目录> runs/<fuse-test目录> --system tsemin
   ```

   生成 `data/predictions/full/tsemin_admin_{de,fr,it}.jsonl.jsonl`（每个 224 行，按编号顺序），并打印 dev / test 子集的分数和 sha256。用官方脚本核对：`--split test` 和 `--split dev` 都应与 fuse 的 summary 一致。
