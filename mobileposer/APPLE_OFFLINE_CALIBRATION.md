# Apple 设备离线 walking calibration

## 目的与边界

入口是 `apple_offline_calibration.py`；计算、数据读取、检测、绘图在
`sensor_apple/offline.py`；多模态时间对齐在 `sensor_apple/alignment.py`；测试在 `../tests/`。
不启动 UDP、不需要连接设备、不加载动作识别/姿态模型，不修改采集 App。

自动流程：读取同一次采集的 NDJSON → 应用时钟映射 → 各模态对齐到共同网格（默认 50 Hz）
→ 在对齐的 IMU 上找到静止/迈步/静止窗口 → 分别计算 `RMI`、`RSB` → 转换数据。
现在包含统一帧率，但**同一采样网格仍不等于跨设备物理时钟已经精确同步**。
完整字段与对时设置见 [多模态对齐说明](APPLE_OFFLINE_ALIGNMENT.md)。

本工具使用 `device_motion` 中的 **userAcceleration（去重力加速度）**、
融合角速度和姿态矩阵；不能以含重力的 raw accelerometer 直接替代。
可选 AirPods 对应 `head_motion`。额外对齐独立 raw 加速度/角速度/磁力计与 UWB；
它不处理 WAV/GPS，也不产生模型预测或人体骨架。缺失的 raw 角速度不以融合角速度冒充。

## 采集约定

1. 默认 Watch 戴左腕、iPhone 放右口袋/右大腿；开始后固定设备，不再调整佩戴。
2. 面向前方站立，保持 N-pose（双臂自然下垂）约 3 秒。
3. 向正前方迈一步，站稳并回到同样的 N-pose；再静止约 3 秒。
4. 开始后续活动。校准动作与普通活动之间留出清楚的停顿。

程序可以检测运动/静止，但不能单靠这些信号证明“这是 N-pose”或“走的是正前方”。
转身、侧迈、挪动手机、挥手、重启 CoreMotion 参考系均会破坏假设。
一个固定 `RSB` 也不能补偿后续手机在口袋中持续滑动。

## 运行

```bash
conda activate mobileposer
cd /Users/zhanglb/WorkSpace/research/project/MocapPipe/mobileposer

# 推荐：一次采集的子文件夹，自动读取该层的 *.ndjson，不递归遍历其它序列
python apple_offline_calibration.py /path/to/session_folder \
  --fps 50 --output data/offline_calibration/session_001

# 也可以指定分设备文件，或一个旧版合并文件 raw.ndjson
python apple_offline_calibration.py /path/to/iphone-events.ndjson \
  /path/to/apple_watch-events.ndjson --output data/offline_calibration/session_002

# 可选第三设备；指定 source-slot 时需要写全本次要用的设备
python apple_offline_calibration.py /path/to/session_folder \
  --source-slot apple_watch:0 --source-slot iphone:3 --source-slot airpods:4 \
  --output data/offline_calibration/session_003
```

需要 NumPy、SciPy、Matplotlib、PyTorch；现有 `mobileposer` 环境已经包含。
输出目录必须不存在或为空；**不会覆盖原始文件或历史 calibration.json**。
出现多次采集混合时应提供 `--session-id UUID`。缺少任一所选设备会报错，不静默降级。

自动检测默认搜索共同可用起点后的前 45 秒，需要恰好一个合格候选：

- 两端共同静止至少 1.5 秒；加速度 RMS < 0.65 m/s²、角速度 RMS < 0.25 rad/s；
- 中间运动 0.3–6 秒；原始采样断点/窗口边缘缺口不超过 0.15 秒；
- 每端水平位移估计至少 0.08 m；初始姿态分散度 p95 ≤ 10°；
- 前后平均姿态差 ≤ 25°；窗口平移 ±100 ms 后方向变化 ≤ 15°。

这些是初始工程阈值，不是已验证的识别准确率保证。检测失败退出码为 2，仍生成
`calibration_quality.json` 和可生成时的 `diagnostic.png`，但不生成可用校准参数。
如果时间对齐阶段成功，仍保留 `aligned_sensors.npz`，可独立分析未校准的数据。
多个候选时不会任意选一个。可查看图后缩小搜索区间：

```bash
python apple_offline_calibration.py /path/to/session_folder \
  --search-start 8 --search-seconds 15 --output data/offline_calibration/review_001
```

或明确提供三个窗口（单位为相对秒数，相对于所选设备中最晚的第一条 motion 样本）：

```bash
python apple_offline_calibration.py /path/to/session_folder \
  --static-window 1 3 --walk-window 3 6 --post-window 6 8 \
  --output data/offline_calibration/manual_001
```

手动窗口仍需通过信号质量检查，不是跳过检查的强制模式。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `aligned_calibrated.npz` | 主输出：同一时间轴上的 IMU/旋转/UWB、校准后的数组、有效性 mask、校准区间结束后的选择 mask |
| `aligned_sensors.npz` | 未进行空间校准的统一时间轴数据，含 `RIS`/`RSI` 与原始样本配对信息 |
| `alignment.json` | 时钟模型、网格 FPS、过滤方法、有效帧比例、UWB 保持策略 |
| `native_modalities.npz` | 所选模态的原始时间戳/序号/采样率副本，UWB 原始样本完整保留；motion 数值已按仓库约定转换坐标/单位 |
| `calibration.json` | 兼容原仓库的 7×3×3 `RMI`/`RSB`、槽位、sessionID、窗口、偏移、坐标约定；未用槽位为单位矩阵，使用时必须结合 `source_slots` |
| `calibration_quality.json` | 检测结果、输入文件/采样率/缺口、静止程度、位移、方向敏感性、失败原因、可选历史对比 |
| `diagnostic.png` | 各设备加速度/角速度 RMS、阈值、窗口；粉色虚线仅标记历史参考窗口 |
| `iphone-calibrated.npz` | 手机最终静止窗口结束以后的转换数据，保留手机原始采样率 |
| `apple_watch-calibrated.npz` | 手表对应数据，保留手表原始采样率 |

下面仅介绍补充输出 `iphone-calibrated.npz` / `apple_watch-calibrated.npz` 的字段。
主输出的 `[T,S,...]` 字段见 [对齐说明](APPLE_OFFLINE_ALIGNMENT.md)。

| 字段 | 形状/单位 | 含义 |
| --- | --- | --- |
| `timestamp_unix_ns` | `[N] int64`，ns | 原始绝对时间戳，不转为浮点纳秒 |
| `timestamp_monotonic_s` | `[N] float64`，s | 原设备单调时间，缺失时为 NaN |
| `timestamp_corrected_s` | `[N]`，s | 原始 Unix 秒 + 显式指定的设备偏移；默认偏移 0 |
| `sequence_number` | `[N] int64` | 原采样序号 |
| `acc_M_m_s2` | `[N,3]`，m/s² | 模型坐标中的去重力加速度 |
| `ori_MB` | `[N,3,3]` | 骨骼 B → 模型 M 的旋转矩阵 |
| `gyro_S_rad_s` | `[N,3]`，rad/s | 经仓库坐标约定转换后的传感器角速度 |
| `gyro_B_rad_s` | `[N,3]`，rad/s | 骨骼坐标中的角速度 |
| `gap_before` | `[N] bool` | 此样本与前一输出样本间隔是否超过 max-gap；没有填补缺口 |
| `source/session_id/slot` | 标量 | 数据源/采集 UUID/身体槽位 |

```python
import numpy as np
x = np.load("data/offline_calibration/session_001/iphone-calibrated.npz")
print(x["acc_M_m_s2"].shape, x["ori_MB"].shape)
```

默认网格校准保存 `grid_indices`、网格纳秒时间戳、完整时钟模型，并在对齐数组中保留
原始插值样本对的时间戳/序号；使用 native 模式时保存原始样本序号，便于准确复现。

## 坐标与时间

完全保留当前 `sensor_apple/sensor.py` 的约定，而不是重新定义 Apple 坐标：

```text
C = diag(1, 1, -1)
RIS = C @ transpose(CoreMotion rotationMatrix) @ C
aS = C @ (userAcceleration * 9.80665)
gyroS = -C @ CoreMotion rotationRate
aI = RIS @ aS
aM = RMI @ aI
RMB = RMI @ RIS @ RSB
gyroB = transpose(RSB) @ gyroS
```

`RXY` 表示 Y → X；`RSB` 是 B → S，不是 S → B。
模型 +Z 取迈步的水平方向。积分沿用在线版本的右端点规则和
`p_corrected = p - 0.5*T*v_end`，依赖起止静止；这是估计短时朝向，不是长期位移追踪。

注意原 `sensor.py` 的 `APPLE_I_UP_RH` 实际取 `[0,0,+1]`，其附近关于 `-Z` 的注释
与常量不一致。本实现记录并沿用实际常量，**没有借本任务改动在线坐标约定**；
与历史参数一致不等于已经独立验证物理轴方向。

默认按共同网格时间间隔积分，向量降采样前低通、随后线性插值；旋转使用 SLERP。
检测阶段仍使用 20 Hz 平滑能量曲线，仅用于检测，与输出 FPS 无关。
可用 `--calibration-sampling native` 恢复逐设备原生采样积分，仍生成统一时间轴输出。
native 模式优先使用设备单调时间间隔，缺少时回退到相对 Unix 时间。
`--clock-offset`、`--clock-drift-ppm` 或 `--sync-anchors` 可校正时钟；程序不把收包延迟
作为时钟偏移，也不把腕部和口袋动作波形自动当作同相同步信号。

本次输出是 NPZ，而不是旧管线的 `processed_30fps.pt`。
原 `apple_postprocess.py` 需要自己的 manifest、输入布局且默认三设备，不能仅换上
此 calibration.json 就假设它完整支持新的两设备目录。

## 与历史参数比较

正常自动运行时，`--reference` **只比较结果**，绝不使用历史窗口、偏移或旋转来帮助检测：

```bash
python apple_offline_calibration.py data/datasets/apple/libo1_0830/seq_03_154502/raw.ndjson \
  --reference data/datasets/apple/libo1_0830/seq_03_154502/calibration.json \
  --output data/offline_calibration/seq03_auto
```

独立的历史回放模式使用保存的窗口、最终 clock offset、所选设备以及最稀疏流最近邻
配对规则；不执行自动检测、不输出 NPZ，也不把历史窗口视为质量检查合格：

```bash
python apple_offline_calibration.py data/datasets/apple/libo1_0830/seq_03_154502/raw.ndjson \
  --reference data/datasets/apple/libo1_0830/seq_03_154502/calibration.json \
  --replay-reference --output data/offline_calibration/seq03_replay
```

指标为两个旋转之间的 SO(3) 测地角（度），不是矩阵逐元素差。
历史校准不是 ground truth；相差小不证明准确，相差大也不能直接判定哪份正确。
旧在线记录没有保存逐窗口的实际时钟偏移/收到的样本列表；最终 offset 可能在后续采集
继续变化，因此历史回放是“按保存信息重建”，不是严格重现当时现场。

第一版 native 校准仅验证过 `seq_01_153854`、`seq_03_154502` 两条真实序列，结果保存在
`data/offline_calibration_audit/20261114-final/`。第二版对齐功能继续使用相同的 seq03 验证，
输出在 `data/offline_alignment_audit/20261114-seq03-50hz-v2/`。这些本地数据/结果不推送远程。

## 测试

从仓库根目录：

```bash
conda activate mobileposer
python -m unittest discover -s tests -v
```

测试包括已知答案的不同采样率双设备序列、真实在线计算函数回归、缺口、无运动、
候选歧义、前后姿态变化、手动窗口长度、分文件读取/重复样本、错误 session/槽位和防覆盖。
