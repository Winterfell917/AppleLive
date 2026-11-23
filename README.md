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

### Apple 离线 walking calibration

无需在线传输：读取一次采集的手机/手表 NDJSON，将 IMU、独立 raw 传感器和 UWB
对齐到共同时间轴（默认 50 Hz），自动寻找“N-pose 静止 → 向前迈步 → 同姿态静止”，
计算每个设备的 `RMI` / `RSB`，输出对齐后的数组及原始采样率副本。
自动检测遇到缺失或歧义会拒绝，不会强行输出参数。

```bash
# 在 mobileposer 目录、mobileposer conda 环境中运行
python apple_offline_calibration.py /path/to/session_folder \
  --watch-placement right_wrist --phone-placement left_thigh_pocket \
  --fps 50 --output data/offline_calibration/my_session
```

支持 Watch 左/右腕与 iPhone 左/右手、左/右大腿口袋的全部 8 种组合，含同侧两设备。
位置需显式声明，每个校准段内安装/握法固定；同段中途换位置需重新校准。
手机手持使用腕部坐标近似并明确标注，不等价于前臂传感器，也不代表现有模型适配所有组合。
参数按设备保存，避免同一身体槽位下相互覆盖；原始时间对齐/UWB 策略不变。
不传位置参数的旧命令仍默认 Watch 左腕、iPhone 右口袋并提示。输入支持旧版合并 NDJSON 和新版分设备文件；
原文件不修改。详细参数、输出字段、历史结果对比及局限见
[离线校准说明](mobileposer/APPLE_OFFLINE_CALIBRATION.md)。
统一时间轴、`RIS/RSI`、时钟偏移/漂移、缺失 mask、UWB 有效期和 NPZ 字段说明见
[多模态对齐说明](mobileposer/APPLE_OFFLINE_ALIGNMENT.md)。同一网格不等于设备已经精确对时。

## Apple 离线动捕：从 RMI、RSB 到人体视频

目标：用 **左腕 Apple Watch + 右大腿口袋 iPhone** 保存的数据，生成全身动作的人体网格视频。
整个过程在电脑上离线完成，不需要实时 UDP、AirPods、Unity，也不用重新训练模型。
采集 App 见 [SmartApple_APP](https://github.com/zhanglbthu/SmartApple_APP)。

先记住两个参数：

| 参数 | 解决什么问题 | 怎么得到 |
| --- | --- | --- |
| RMI | 每台设备的“世界坐标系”不一定相同，需要转换到模型的世界坐标系 | 把设备按已知方向放好，静止采一段数据 |
| RSB | 设备的方向不等于手腕或大腿的方向，需要校准佩戴角度 | 戴好设备，站立保持 N-pose |

手机和手表各有一套 RMI、RSB，不共用同一个矩阵。

```text
按已知方向静止放置设备 → 计算并保存 RMI
戴好设备，先站立静止，再正常运动 → 用开头的 N-pose 计算 RSB
按时间戳对齐数据 → 转换加速度和姿态 → 输入模型 → 生成 MP4
```

### 1. 准备环境和数据

先按上面的“安装”部分准备环境。**本节命令都在仓库根目录运行**，不要先进入 mobileposer 子目录。

```bash
conda activate mobileposer
```

确认以下两个文件存在：

```text
mobileposer/data/checkpoints/base_model_12combo.pth   # 动作估计模型
mobileposer/smpl/basicmodel_m.pkl                    # SMPL 人体模型
```

每次采集结束后，先等手表文件传到手机，再把该次采集的整个文件夹复制到电脑。
文件夹至少需要包含：

```text
iphone-events-<sessionID>.ndjson
apple_watch-events-<sessionID>.ndjson
```

两份文件必须属于同一次采集（sessionID 相同）。音频、回执等文件可以保留，程序不会把它们作为动捕模型输入。
下面命令中的 `/path/to/...` 都要替换成自己的数据目录。

### 2. 先计算 RMI：已知方向的静止采集

1. 选定模型世界坐标：+Y 向上，+Z 是你指定的前方，+X 按右手坐标系确定。
2. 把手机和手表分别按**已知的物理方向**放好。它们可以朝向不同，但各自的轴向必须明确。
3. 点击开始后，等待操作造成的晃动结束，再静止保持约 8–10 秒。整段录制 12–20 秒更方便排除首尾操作。
4. 将该序列复制到电脑，填写固定姿态配置，再运行：

```bash
python mobileposer/apple_static_rmi.py /path/to/static_session \
  --pose-config mobileposer/configs/apple_static_rmi_pose.json \
  --window-seconds 8 \
  --output mobileposer/data/static_rmi/my_static_pose
```

配置文件里的 `RMS_target s` 分别描述手机和手表的已知方向。
每个矩阵的三列，是**经过程序坐标转换后的传感器 +X、+Y、+Z 轴在模型世界坐标系中的方向**。
不能仅凭“设备静止”自动猜出这个矩阵；静止本身不能确定朝向。

**注意：仓库自带的配置用于复现之前的草图实验，不是所有 Apple 设备通用的摆放规则。**
该实验的重力方向检查没有通过。要复现实验可保留它；要做严格验证，应先确认物理轴向，再修改配置。

程序会自动选择两台设备都比较稳定的 8 秒，主要输出：

- `calibration.json`：每台设备的 RMI、检查结果和时间区间。
- `diagnostic.png`：静止区间和运动信号，方便检查有没有选到晃动。
- `candidate_rmi.npz`、`pose_config_snapshot.json`：矩阵及本次使用的配置。

也可以用 `--window START END` 手动指定 5–10 秒的区间。
时间从**两端最早的一条 device_motion 样本**开始计，不是从按按钮开始计。

如果显示 `blocked_coordinate_validation` 或返回码为 2，先看输出目录里的检查结果：
可能已经保存了候选矩阵，但方向检查没有通过。不要把“生成了文件”理解成“校准正确”。

### 3. 再采动作：开头保持 N-pose

手表戴在**左腕**，手机放在**右大腿口袋**，整段过程中不要换位置或改变佩戴方向。

点击开始后：

1. 面向之前选定的模型 +Z 方向，站直、双臂自然下垂、手腕放松，这就是这里的 N-pose。
2. 等点击按钮造成的动作结束，静止保持几秒，尽量在开头 8 秒内留出至少 2 秒稳定区间。
3. 开始正常运动，结束后把两端数据复制到电脑。

不需要另开程序单独计算 RSB；下一步会自动从这段序列的开头计算。

### 4. 计算 RSB、运行模型并生成视频

下面的命令用于复现我们已经看过效果的流程：使用保存的 RMI，从动作序列的第 4–6 秒计算 RSB，再运行模型。

```bash
python mobileposer/apple_static_mocap.py \
  --session /path/to/mocap_session \
  --rmi mobileposer/data/static_rmi/my_static_pose/calibration.json \
  --config mobileposer/configs/apple_static_lw_rp.json \
  --npose 4 6 \
  --allow-unverified-rmi \
  --output mobileposer/data/static_rmi_eval/my_mocap
```

几个参数需要注意：

| 参数 | 含义 |
| --- | --- |
| `--session` | 动作序列文件夹，不是前面的静态序列 |
| `--rmi` | 前面保存的 RMI 文件 |
| `--npose 4 6` | 用第 4–6 秒拟合 RSB；这是之前那段数据的设置，不适合直接套用所有序列 |
| `--allow-unverified-rmi` | 明确允许一次探索性测试，包括跨序列复用或未通过方向检查的 RMI；不会把它标记为验证通过 |
| `--output` | 新的输出目录；已有目录不会被覆盖 |
| `--skip-video` | 只保存计算结果，不渲染视频 |

对于新的序列，可以去掉 `--npose 4 6`，程序会在开头 0.5–8 秒中自动寻找两端共同稳定的 2 秒。
手动指定的区间也会检查稳定性。**程序只能检查是否静止，不能判断你是否真的摆出了 N-pose。**

如果 RMI 已通过内部检查且来自同一次采集，可不加 `--allow-unverified-rmi`。
如果来自不同采集，程序默认会拒绝复用，因为重新初始化后参考系可能变化；同样的参考系编号也不代表方向完全一致。

程序接下来会：

- 将两端的有效重叠数据对齐到 **30 Hz**；姿态做旋转插值，加速度等向量做滤波和插值。
- 不跨越超过 0.15 秒的缺口补数据；必需的运动数据不完整时停止推理。
- 把左腕放入模型槽位 0，右大腿放入槽位 3，其他槽位的加速度和姿态矩阵填零。
- 使用现有模型计算 24 个关节的姿态，再渲染人体网格。

当前模型直接使用的是**去除重力的加速度 + 姿态**，不直接使用角速度、UWB、磁力计或音频。
这些其他模态有数据时可另外分析，但不会自动参与本次动作估计。

### 5. 查看哪些结果

在输出目录中，先看这几个文件：

| 文件 | 用途 |
| --- | --- |
| `mocap_mesh.mp4` | 人体网格视频，包含正面和斜侧面视角 |
| `npose_diagnostics.png` | 检查开头选中的校准区间是否稳定 |
| `calibration.json` | 查看手机、手表各自的 RMI 和 RSB |
| `model_io.npz` | 查看实际模型输入和预测姿态，便于后续分析 |
| `aligned_sensors.npz` | 查看按共同时间轴整理的传感器数据和有效标记 |
| `report.json` | 查看配置、时间对齐、输入文件校验和运行警告 |

完整结果还包括原采样时间的数据数组、PyTorch 结果、配置副本和视频截图。
数组中部分数据已经转换了轴向和单位，**不等于原始文件**；原始 NDJSON 不会被修改，应另行保留。

读取模型输出的简单示例：

```python
import numpy as np

data = np.load("mobileposer/data/static_rmi_eval/my_mocap/model_io.npz",
               allow_pickle=False)
pose = data["pose_local_rotation"]  # [帧数, 24, 3, 3]：关节局部旋转矩阵
acc = data["acc_M_slots"]           # [帧数, 5, 3]：模型世界系中的加速度
ori = data["ori_MB_slots"]          # [帧数, 5, 3, 3]：校准后的骨骼姿态
```

先看 N-pose 结束后的动作：左右是否对应、抬手方向是否正确、是否抖动或出现不合理弯曲。
视频固定人体根部位置，主要看姿态，**不能用来评价真实行走轨迹**。
之前测试得到 545 帧、30 FPS，约 18.2 秒的视频，视觉效果符合预期。

### 6. 简要原理与注意事项

记号约定：`RXY` 表示把 Y 坐标系中的向量转换到 X 坐标系。
M 是模型世界系，I 是设备参考系，S 是转换后的传感器系，B 是骨骼系。

核心计算如下，手机和手表分别计算：

```python
RMI = RMS_target @ mean_RIS_static.T        # 用已知固定方向校准参考系
RSB = (RMI @ mean_RIS_N).T @ target_RMB_N    # 用 N-pose 校准佩戴方向
RMB = RMI @ RIS @ RSB                      # 得到骨骼在模型世界系中的姿态
aM = RMI @ RIS @ aS                        # 得到模型世界系中的加速度
```

这里的平均姿态是通过 SVD 得到的旋转矩阵，不是直接平均欧拉角。
`aS` 已经在传感器系中，所以计算 `aM` 时**不要再额外乘 RSB**。

程序保留了之前实验的坐标转换；不要对已处理的数据重复转换：

```python
import numpy as np

C = np.diag([1, 1, -1])
RIS = C @ Q.T @ C                    # Q 为 NDJSON 中保存的旋转矩阵
RSI = RIS.T
aS = C @ user_accel_g * 9.80665       # 转为 m/s²，已去除重力
```

这只是本项目当前采用的约定，不是在断言 Apple 官方坐标系是左手系。
更详细的时间戳、缺失值和 UWB 处理规则见 [多模态对齐说明](mobileposer/APPLE_OFFLINE_ALIGNMENT.md)。

最后记住三点：

- **动作看起来自然，不等于校准已经严格验证。** N-pose 拟合和模型本身可能掩盖错误；之前保存的 RMI 仍有重力方向检查问题。
- **同一时间网格，不等于设备时钟已经精确同步。** 当前流程默认使用记录中的 Unix 时间，没有独立验证两台设备的物理时间差。
- **换佩戴方式要重新算 RSB；跨初始化复用 RMI 要检查参考系。** 有条件时，用已知姿态、重复采集或参考视频进一步验证。

视频渲染需要 Open3D、OpenGL 图形环境和 H.264 编码支持，SMPL 和权重需遵守各自许可。
若机器不支持渲染，可先加 `--skip-video` 保存计算结果。
录制数据和生成结果放在被 Git 忽略的 `mobileposer/data/` 中，不会随代码上传。

### 7. 对应代码

| 代码 | 作用 |
| --- | --- |
| [apple_static_rmi.py](mobileposer/apple_static_rmi.py) | 静态 RMI 计算入口 |
| [apple_static_mocap.py](mobileposer/apple_static_mocap.py) | RSB 计算、模型推理与视频生成入口 |
| [static_calibration.py](mobileposer/sensor_apple/static_calibration.py) | 校准公式、稳定区间选择与检查 |
| [alignment.py](mobileposer/sensor_apple/alignment.py) | 不同采样率的数据对齐 |
| [mesh_video.py](mobileposer/sensor_apple/mesh_video.py) | 人体网格视频渲染 |
| [静态摆放配置](mobileposer/configs/apple_static_rmi_pose.json) / [动捕配置](mobileposer/configs/apple_static_lw_rp.json) | 固定方向、佩戴位置、帧率等设置 |

检查代码是否正常，可在仓库根目录运行：

```bash
python -m unittest discover -s tests -v
```

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
