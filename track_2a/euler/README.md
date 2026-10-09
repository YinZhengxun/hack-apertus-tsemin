# 在 Euler 上跑

这套代码只调用 CSCS 的推理接口，不需要 GPU。放到 Euler 上跑的好处是不占笔记本、网络快、能挂着跑几个小时。

**关键事实：Euler 的计算节点没有外网**。所以调接口的任务（`check` / `smoke` / `run`）要在**登录节点**上用 `tmux` 跑，不是 `sbatch`。这些任务几乎不吃 CPU（99% 的时间在等网络），放登录节点没问题。`sbatch` 只在两种情况下用：① 做 GPU 实验（D5 / D6）；② `euler/api_job.sh` 里的代理试探确认计算节点能经 `proxy.ethz.ch` 出网。

## 1. 把项目拷过去（Windows 的 PowerShell）

整个 `track_2a` 只有 24 MB，`runs/` 和 `__pycache__` 不用带：

```powershell
cd D:\apertus\uzh-semantic-diff
tar --exclude=track_2a/runs --exclude=__pycache__ -czf $env:TEMP\track_2a.tgz track_2a
scp $env:TEMP\track_2a.tgz zheyin@euler.ethz.ch:~/
```

然后在 Euler 上解开（会覆盖同名旧文件，`runs/` 不受影响）：

```bash
mkdir -p ~/uzh-semantic-diff && tar -xzf ~/track_2a.tgz -C ~/uzh-semantic-diff
sed -i 's/\r$//' ~/uzh-semantic-diff/track_2a/euler/*.sh          # 去掉 Windows 换行
printf 'LLM_API_KEY=<your key>\n' > ~/uzh-semantic-diff/track_2a/.env
```

拷回结果（在 Windows 上）：

```powershell
scp -r zheyin@euler.ethz.ch:uzh-semantic-diff/track_2a/runs/<运行目录> D:\apertus\uzh-semantic-diff\track_2a\runs\
```

## 2. 登录节点上先自检（几秒钟）

```bash
cd ~/uzh-semantic-diff/track_2a
python3 --version            # 登录节点自带 3.10，够用，不装任何包
python3 run.py selftest
python3 run.py check
```

`check` 通过 = 登录节点连得上接口。如果报 `No module named 'uzh_diff'`，说明 `src/` 没拷全，重做第 1 步。

## 3. 正式跑：登录节点 + tmux

每个方法开一个 tmux 窗口，互不干扰：

```bash
cd ~/uzh-semantic-diff/track_2a && mkdir -p runs
tmux new -d -s m2  'python3 run.py run --method m2  --manifest dev60 --workers 2 2>&1 | tee runs/tmux_m2.log'
tmux new -d -s d4  'python3 run.py run --method d4  --manifest dev60 --workers 2 2>&1 | tee runs/tmux_d4.log'
tmux new -d -s b0s 'python3 run.py run --method b0s --manifest dev60 --workers 2 2>&1 | tee runs/tmux_b0s.log'
```

看进度 / 收尾：

```bash
tmux ls                                  # 还在跑的会话；跑完的会自动消失
tail -n 20 runs/tmux_m2.log              # 看日志（不进会话）
tmux attach -t m2                        # 进会话看实时输出；Ctrl+B 再按 D 退出，任务继续跑
ls -dt runs/*/ | head                    # 结果目录，最新的在最前
```

两点注意：

- **tmux 会话绑在你登录的那台登录节点上**（提示符里的 `eu-login-11` 之类）。下次 `ssh zheyin@euler.ethz.ch` 可能被分到别的节点，`tmux ls` 就看不到。先记下节点名，不在同一台时用 `ssh eu-login-11` 跳过去再 `tmux attach`。
- 中途断了直接重跑同一条命令：`runs/_cache/` 让已经问过的请求不再发，从断点接着跑。

跑完后的分析不联网，在哪台都能做：

```bash
python3 run.py analyze --run runs/<m2目录> --d4-run runs/<d4目录>
python3 run.py compare --run runs/<m2目录>
```

## 4. 可选：试一下计算节点能不能经 ETH 代理出网

`euler/api_job.sh` 会设置 `https_proxy=http://proxy.ethz.ch:3128`，先花 20 秒探一次接口，通了才跑命令，不通就立刻退出并写明原因。只需提交一次短作业试探：

```bash
cd ~/uzh-semantic-diff/track_2a
sbatch --time=00:10:00 euler/api_job.sh check
squeue -u zheyin
cat uzhdiff-<作业号>.out
```

日志里出现 `api reachable` 就说明以后可以 `sbatch euler/api_job.sh run --method m2 --manifest dev60 --workers 4` 用计算节点跑；出现 `api unreachable` 就继续用第 3 节的 tmux，不用再试。

## 5. GPU（只有做隐状态探针 D5 / D6 时才需要）

计算节点无外网，权重要先在登录节点下载到 scratch，作业里离线加载。作业头照这个写（卡型必须写名字，否则会落到没有 bf16 的 Quadro RTX 6000）：

```bash
#!/bin/bash -l
#SBATCH --account=es_chatzi
#SBATCH --gpus=nvidia_a100_80gb_pcie:1        # 备选 nvidia_a100-pcie-40gb:1；8B bf16 推理 24 GB 的 4090/3090 也够
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --time=06:00:00
#SBATCH --output=/cluster/scratch/zheyin/logs/%x_%j.log
[ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
source /cluster/scratch/zheyin/modules.env
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}; mkdir -p "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
source /cluster/scratch/zheyin/venv/vllm/bin/activate
```

下载权重（登录节点，需要先在 Hugging Face 同意 `swiss-ai/Apertus-v1.5-8B` 的使用条款并 `huggingface-cli login`）：

```bash
huggingface-cli download swiss-ai/Apertus-v1.5-8B --local-dir /cluster/scratch/zheyin/models/Apertus-v1.5-8B
```

`gpu_job.sh` 和对应的脚本在确定要做 D5 / D6 之后再写。scratch 里 15 天没动的文件会被自动删，权重放进去后要么用起来，要么定期 `touch`。
