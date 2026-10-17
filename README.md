# MocapPipe

基于少量消费级 IMU 的实时全身动作捕捉项目。当前精简版保留：

- MobilePoser 核心姿态、关节、足部接触和速度模型；
- Apple Watch + iPhone（可选 AirPods）实时接入；
- 华为设备实时接入；
- CHI 2027 `ours`/`plain` 与 TIC 在线 IMU 校准器；
- SMPL 人体模型与 Unity MotionViewer 输出；
- 实时运行所需的预训练权重。

## 安装

推荐 Python 3.9：

```bash
conda create -n mobileposer python=3.9
conda activate mobileposer
pip install -r requirements.txt
pip install -e .
```

实时脚本采用项目原有的顶层导入方式，请从 `mobileposer` 目录运行：

```bash
cd mobileposer
```

## Apple 设备动作捕捉

### 设备和槽位

Sensor Read 的默认设备映射为 Watch + iPhone。运行时不会等待 AirPods；
未配置的头部槽位以零值填入。

| 数据源 | 身体位置 | 模型槽位 | 默认 |
| --- | --- | ---: | --- |
| Apple Watch | 左腕 | 0 | 必选 |
| iPhone | 右口袋/右大腿 | 3 | 必选 |
| AirPods | 头部 | 4 | 可选 |

默认命令只等待 Watch 和 iPhone：

```bash
python livedemo_apple.py --no-viewer --duration 20
```

需要耳机时显式加入头部槽位：

```bash
python livedemo_apple.py \
  --source-slot apple_watch:0 \
  --source-slot iphone:3 \
  --source-slot airpods:4
```

无耳机时可以完成实时推理，但精度和稳定性通常低于 Watch + iPhone + AirPods
三设备配置。两设备模式只支持 `none`/`nocalibration`；`ours`、`plain`、`tic`
以及 `--compare-ours-nocalibration` 均要求槽位 0、3、4 同时存在。

这里的 `nocalibration` 只表示不使用学习式在线 calibrator，默认的
`walking_6dof` 物理坐标校准仍会执行。如需连物理校准也跳过，请额外传入
`--calibration none`。

### Sensor Read 和基本运行

在 Sensor Read 中将接收地址设置为运行 Python 的电脑局域网 IP，UDP 端口设置为
`9000`（脚本的默认 `--port`）。启动各设备采集后，从 `mobileposer` 目录运行。

先测试无可视化链路：

```bash
python livedemo_apple.py --no-viewer --duration 20
```

运行带轻量火柴人预览的实时演示：

```bash
python livedemo_apple.py
```

使用新版 CHI 2027 calibrator：

```bash
# 单独运行 Calibration 分支
python livedemo_apple.py --calibrator ours

# 同时计算 NoCalibration 和 Calibration；GTX 1650 建议 25 FPS
python livedemo_apple.py --compare-ours-nocalibration --fps 25
```

比较模式维护两份独立的 MobilePoser 时序状态。保存文件中 `pose` 和
`pose_ours` 为 Calibration 输出，`pose_nocalibration` 为未经过学习式
calibrator 的输出。

### Unity 可视化

Unity MotionViewer 使用的连接与 Sensor Read UDP 端口无关。当前 Python
MotionViewer 是 TCP 服务端，固定配置为：

| Unity 设置 | 值 |
| --- | --- |
| Server/Host | `127.0.0.1` |
| Port | `8989` |
| Protocol | TCP |

Unity 和 Python 需要运行在同一台电脑，因为服务端绑定的是本机回环地址。
推荐启动顺序：

1. 在 Unity 中打开 MotionViewer 场景，将 Host 设置为 `127.0.0.1`、Port 设置为 `8989`，暂不连接；
2. 运行下面的 Python 命令，等待终端出现 `Waiting for unity3d to connect`；
3. 在 Unity 中进入 Play 模式或点击连接。

单路 Unity 可视化：

```bash
python livedemo_apple.py --viewer unity --calibrator nocalibration
```

双路 Unity 对比：

```bash
python livedemo_apple.py \
  --viewer unity \
  --compare-ours-nocalibration \
  --fps 25
```

双路模式中，Unity 的两个角色标签分别为 `NoCalibration` 和 `Calibration`。
如果出现端口占用错误，请先关闭其他 MotionViewer/Python 进程；如需改端口，需让
Unity 客户端配置与 `mobileposer/articulate/utils/unity/view_motion.py` 中的
`MotionViewer.port` 保持一致。

### 数据采集

按 subject/name 采集一段数据：

```bash
python livedemo_apple.py --subject libo1_0901 --name 001
```

数据保存到 `mobileposer/data/datasets/apple/libo1_0901/001/`。默认骨架预览
在独立进程中只计算24个关节，不生成SMPL mesh，不会等待显示队列。

详见 [Apple 接入说明](mobileposer/APPLE_MOCAP.md)。

### 训练数据：Watch 本地记录模式

训练采集默认不再等待 Watch 实时 UDP。iPhone 和 Apple Watch 在每条 take 中
分别本地记录；标定阶段 iPhone 同时通过 UDP 发送 `device_motion`，让电脑在正式
动作前立即检查标定。标定通过后关闭 iPhone UDP，但两台设备继续本地记录；结束后
Watch 将完整文件传给手机。先运行一次试采：

```bash
python collect_train_sequence.py --once \
  --action Walking --watch lw --phone-pocket lp \
  --group 1 --phone-facing screen_in
```

脚本负责提示 N-pose、向前走和正式动作的时间窗口，并先生成 `capture.json`。
开始每条采集前，将手机 UDP 地址设置为电脑局域网 IP、端口 `9000`，打开 UDP
实时传输。脚本只等待 iPhone，不等待 Watch；标定失败会立即要求重做，成功后会
提示关闭手机 UDP，再开始正式动作。`--host`/`--port` 可修改电脑监听地址和端口。
每条结束后，它会显示该条专属的电脑收件箱；等 Watch 文件完整传到手机后，
通过 USB 将 iPhone 和 Watch 的 NDJSON 复制进该目录，然后回到终端按回车。
脚本会自动识别两份文件、生成 `labels.pt`，成功后才进入下一条。
整理时只使用 iPhone 的 N-pose 检查和向前走数据计算 `R_MI`；Apple Watch
不单独标定，直接使用同一个 iPhone `R_MI`。iPhone 标定失败时，该 attempt
会记录失败原因，脚本要求重新标定，不会开始正式动作。文件导入时还会使用本地
iPhone 文件和成功标定窗口重新校验一次，避免 UDP 丢包或时间偏差造成误判。

`labels.pt` 保存设备顺序、文字槽位、整数 `model_slots` 槽位真值，以及 30 FPS
模型坐标系序列：`orientation = R_MI @ RIS`、
`acceleration = R_MI @ RIS @ aS`。同时保存 `motion_valid` 新鲜度掩码；不再保存
`R_SB`、`R_MB`、原始 `aS`/`RIS` 或 UWB 序列。共享的 `R_MI` 及手机标定质量
记录在 `meta.json` 中。

可以用 `--inbox D:/MocapInbox` 更改收件箱根目录。如果本次只想采集、稍后统一
整理，添加 `--defer-import`。之后手动执行：

```bash
python collect_train_sequence.py \
  --finalize data/datasets/train_labels/_trial/<试采目录> \
  --iphone-file D:/raw/iphone.ndjson \
  --watch-file D:/raw/watch.ndjson
```

成功后目录中会生成 `labels.pt`、`meta.json`、`quality.json` 和合并后的
`raw.ndjson`。如果设备文件时间与电脑标记窗口存在固定偏差，可分别使用
`--iphone-time-offset` 和 `--watch-time-offset`（秒）修正。旧版双设备实时 UDP
流程仍可通过 `--legacy-live` 显式启用。

## 华为设备动作捕捉

启动传感器发送端和 Unity MotionViewer 后运行：

```bash
python livedemo.py --mocap
```

三路校准对比：

```bash
python livedemo.py --mocap --compare-all
```

## 运行权重

```text
mobileposer/data/checkpoints/
├── base_model_12combo.pth
├── chi2027_calibrator_ours/
│   └── best.pt
├── chi2027_calibrator_plain/
│   └── best.pt
├── combo_imu_calibrator_lw_rp_h_ori_only_jerk_nopose_fulltrain_tb_noncausal/
│   └── best.pt
└── tic_calibrator_amass_full/
    └── best.pt
```

`smpl/basicmodel_m.pkl` 是实时姿态输出所必需的 SMPL 模型文件。

## 目录结构

```text
mobileposer/
├── livedemo.py             # 华为实时入口
├── livedemo_apple.py       # Apple 实时入口
├── sensor_huawei/          # 华为 UDP 与标定
├── sensor_apple/           # Sensor Read UDP 与标定
├── models/                 # 推理模型与在线校准器
├── articulate/             # SMPL 数学与 Unity MotionViewer
├── utils/model_utils.py    # 权重加载与姿态补全
├── smpl/                   # SMPL 模型
└── data/checkpoints/       # 运行权重
```

## 性能提示

当前 MobilePoser 的小批量 Packed LSTM 在 Apple Silicon CPU 上明显快于项目所用
PyTorch 2.1 MPS 后端，因此 `livedemo_apple.py --device auto` 在 Mac 上默认选择 CPU。

## License

本项目基于 MobilePoser，遵循仓库中的 [LICENSE](LICENSE)。
