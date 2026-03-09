# TensorBoard → W&B 实时转写同步（不依赖 Isaac 内部 wandb）

适用场景：
- 训练进程（Isaac Sim/OmniIsaacGymEnvs）只写 TensorBoard `tfevents`。
- 另起一个独立 conda/venv（例如 `wandb_sync`）读取 `tfevents` 并用 `wandb.log()` 上传到 W&B。
- 通过“转写”把指标名从 `Mar05_.../Episode/x` 规范化为 `Episode/x`（去掉目录前缀）。

脚本位置：
- [tools/wandb_tb_rewrite_sync.py](../tools/wandb_tb_rewrite_sync.py)

---

## 1. 准备独立环境（wandb_sync）

```bash
conda create -n wandb_sync python=3.10 -y
conda activate wandb_sync
python -m pip install -U wandb tensorboard
wandb login
```

> 注意：这个环境与 Isaac Sim 自带 Python 完全隔离，不会触发 pydantic 版本冲突。

---

## 2. 找到 TensorBoard logdir

你的工程会把日志写到类似：

- `.../OmniIsaacGymEnvs/runs/<EXP>/<TIME_STR>/<WRITER_TIME>/events.out.tfevents.*`

其中 **logdir** 是包含 `events.out.tfevents.*` 的目录，例如：

- `/home/loop/isaac_sim-2023.1.1/OmniIsaacGymEnvs/runs/USV/Mar05_16-07-46/Mar05_16-08-18`

---

## 3. 启动“常驻同步”

在另一个终端（不要影响训练终端）里：

```bash
conda activate wandb_sync

python tools/wandb_tb_rewrite_sync.py \
  --tb-logdir "/home/loop/isaac_sim-2023.1.1/OmniIsaacGymEnvs/runs/USV/Mar05_16-07-46/Mar05_16-08-18" \
  --entity loopzhang7-zhejiang-university \
  --project OmniIsaacGymEnvs \
  --run-name Mar05_16-08-18 \
  --strip-prefix auto \
  --poll-interval 10
```

说明：
- `--strip-prefix auto`：自动使用 `basename(tb-logdir)/` 作为可剥离前缀（例如 `Mar05_16-08-18/`）。
- 默认只同步 `Episode/`、`Loss/`、`rewards/`、`Diagnostics/` 四类标量曲线。
- 脚本会在 `tb-logdir` 下写一个 state 文件（`.wandb_tb_rewrite_state.<run_name>.json`），用于断点续传、避免重复上传。

---

## 4. 常见可选参数

### 固定 run id（推荐）
如果你希望重启脚本也始终写入同一个 W&B run：

```bash
python tools/wandb_tb_rewrite_sync.py ... --run-id usv-mar05-160818
```

不指定 `--run-id` 也可以：脚本会自动生成并写入 state 文件，后续自动复用。

### 手动 strip-prefix
当你的 TB tag 本来就带了某个前缀（不一定等于目录名），可以手动指定：

```bash
--strip-prefix "Mar05_16-08-18/"
```

或禁用：

```bash
--strip-prefix none
```

### 自定义同步范围
默认同步四类前缀；你也可以自定义：

```bash
--include-prefix Episode/ \
--include-prefix Loss/
```

---

## 5. 守护运行（tmux 推荐）

```bash
tmux new -s wb_rewrite
conda activate wandb_sync
python tools/wandb_tb_rewrite_sync.py ...
```

退出但不停止：`Ctrl+b` 然后 `d`

---

## 6. 验证与排障

- 先干跑（不上传）确认脚本能读到 scalars：

```bash
python tools/wandb_tb_rewrite_sync.py ... --dry-run
```

- 若长时间显示 `idle (no new scalars)`：
  - 确认训练仍在写入 tfevents（文件大小是否增长）
  - 确认 `--tb-logdir` 指向“包含 tfevents 的目录”
  - 确认 TB 的 tag 前缀确实是 `Episode/`、`Loss/`、`rewards/`、`Diagnostics/`（否则放宽 include）
