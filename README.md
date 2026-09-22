# MocapPipe

基于少量消费级 IMU 的实时全身动作捕捉项目。当前精简版保留：

- MobilePoser 核心姿态、关节、足部接触和速度模型；
- Apple Watch + iPhone + AirPods 实时接入；
- 华为设备实时接入；
- ComboTemporal 与 TIC 在线 IMU 校准器；
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

Sensor Read 默认映射：

| 数据源 | 身体位置 | 模型槽位 |
| --- | --- | ---: |
| Apple Watch | 左腕 | 0 |
| iPhone | 右口袋/右大腿 | 3 |
| AirPods | 头部 | 4 |

先测试无可视化链路：

```bash
python livedemo_apple.py --no-viewer --duration 20
```

运行带轻量火柴人预览的实时演示：

```bash
python livedemo_apple.py
```

按 subject/name 采集一段数据：

```bash
python livedemo_apple.py --subject libo1_0901 --name 001
```

数据保存到 `mobileposer/data/datasets/apple/libo1_0901/001/`。默认骨架预览
在独立进程中只计算24个关节，不生成SMPL mesh，不会等待显示队列。

详见 [Apple 接入说明](mobileposer/APPLE_MOCAP.md)。

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
