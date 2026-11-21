# 离线手机 + Watch：多模态共同时间轴与空间校准

## 一条命令

激活 `mobileposer` Conda 环境，在仓库的 `mobileposer` 目录运行：

```bash
python apple_offline_calibration.py /path/to/session_folder \
  --watch-placement right_wrist --phone-placement left_thigh_pocket \
  --fps 50 --output data/offline_calibration/session_001
```

输入可以是一个序列文件夹（只读取该层 NDJSON，不递归其它序列），一个旧版合并
`raw.ndjson`，或依次指定手机和手表两个 NDJSON。支持中文/空格路径（命令行加引号）。
输出目录必须为空。App 和原始记录不改变，不需要联网、UDP、模型权重或真机。

可选 Watch 左/右腕，手机左/右手或左/右大腿口袋，包含同侧戴表并持手机的组合。
用 `--watch-placement` + `--phone-placement` 明确声明（完整选项与采集姿态见
[位置说明](APPLE_OFFLINE_CALIBRATION.md)）。旧命令未传位置时
仍假定 Watch 左腕 slot 0、手机右口袋 slot 3，报告会警告；不自动猜测佩戴位置。
顺序以输出 `source_names` 为准，不要假设第 0 列是手机。
高级 `--source-slot` 可添加 AirPods，但物理位置标签记为 `unspecified`，不能与命名位置混用。

位置每段可以不同，同段内必须保持安装/握法不变。改变位置选项影响目标 N-pose、
`RSB` 及身体坐标变换，不改变原始 `RIS/RSI`、时间映射、重采样或 UWB 距离。
RMI 由向前运动估计，与位置槽位选择无关；两设备仍分别估计自己的 RMI。

流程：**读取 → 时钟映射 → 统一网格 → 校准区间检测 → RMI/RSB → 转换全部网格数据**。
输出包含校准片段；`after_calibration` 标记校准结束后的运动，不默认丢掉前段。

## RSI、RIS、RMI、RSB 的区别

沿用仓库的 `RXY` 记号：将 Y 坐标系的向量转换到 X 坐标系。

- `RIS[t]`：传感器 S → 惯性 I，来自 CoreMotion，每帧变化。
- `RSI[t] = RIS[t].T`：惯性 I → 传感器 S，每帧变化；**不是设备到身体的固定安装参数**。
- `RMI`：惯性 I → 模型 M，walking calibration 的固定输出。
- `RSB`：身体部位 B → 传感器 S，walking calibration 的固定安装参数。
- `RMB[t] = RMI @ RIS[t] @ RSB`：身体部位 → 模型，输出为 `ori_MB`。

本实现同时提供 `RIS`、`RSI`、`RMI`、`RSB`，避免将 `RSI` 与 `RSB` 混用。
这里的旋转矩阵不是独立采集的新传感器模态；它们是姿态或其变换。

## 时间映射：不能把重采样当成精确对时

默认使用各设备保存的 Unix 时间戳，不使用收到文件/收到 UDP 包的时间。
设备单调时间戳的起点不同，不能直接互相相减对齐。
默认报告明确标记 `clock_sync_accuracy_verified: false`；程序无法仅凭这些数据
保证两块设备物理时钟一致，也不会把手臂和大腿的真实运动相位差“校正”掉。

默认映射：`t_common = t_raw`。如已独立测得偏移/漂移，可指定：

```bash
python apple_offline_calibration.py /path/to/session_folder \
  --clock-offset apple_watch:0.012 --clock-drift-ppm apple_watch:8 \
  --fps 50 --output data/offline_calibration/clock_corrected
```

正偏移表示向时间戳加秒数；正 drift 表示拉长时间间隔。完整公式：

```text
t_common = t_origin + (t_raw - t_origin) * (1 + drift_ppm * 1e-6) + offset_s
```

`t_origin` 为所有所选 motion 流中最早的第一帧原始 Unix 时间，精确纳秒值写入报告。
同一个设备的 motion、raw IMU、UWB **统一应用同一时钟模型**，原始时间戳始终保留。

也可提供人工确认的“同一物理事件”锚点 JSON：

```json
{
  "apple_watch": [
    {"source_unix_ns": "1788000000000000000", "reference_unix_ns": "1788000000100000000"},
    {"source_unix_ns": "1788000100000000000", "reference_unix_ns": "1788000100102000000"}
  ]
}
```

```bash
python apple_offline_calibration.py /path/to/session_folder \
  --sync-anchors /path/to/anchors.json --output data/offline_calibration/anchored
```

`reference_unix_ns` 必须在同一参考时间域（例如 iPhone 原始时间域）；这些数字仅为格式示例。
单个锚点只估计 offset；多个锚点拟合 offset + drift。多个锚点需跨至少 1 秒，
拟合残差超过 20 ms、漂移超过 10000 ppm 会报错。拟合残差不是实际同步精度。
同一设备不能同时使用 anchors 和手动 offset/drift。

**不要使用上次为拟合历史 RMI 反推的 0.656/1.099 秒作为新序列的对时参数。**
它们不是通用时钟校准值。

## 各模态的处理

| 模态 | 对齐方法 | 缺失/异常处理 |
| --- | --- | --- |
| motion 去重力加速度、融合角速度 | 原始时间戳 → 向量低通（需要降采样时）→ 线性插值 | 不外推、不跨越 >150 ms 缺口，输出 NaN + mask |
| 姿态 `RIS/RSI` | SO(3) SLERP；不对欧拉角直接插值 | 同样的时间覆盖/缺口门限，旋转没有额外抗混叠滤波 |
| 独立 raw 加速度/角速度/磁力计 | 各自时间戳分别重采样 | 没有该流则全 NaN + false，不用融合值冒充 raw |
| UWB 距离 | 最近一条**过去**的测量保持，默认最多 0.3 秒 | 无未来值借用；超时、负距离、无距离值时无效 |

默认 `--anti-alias auto`：原流速率 >1.1 倍目标 FPS 时，向量使用 4 阶 Butterworth
双向零相位低通，截止频率为目标 FPS 的 0.4 倍。按中位采样间隔近似采样率，
不跨缺口滤波；少于 32 样本的片段不滤波并记录。`--anti-alias none` 可关闭。
这是向量重采样的预处理，原始样本副本不受影响。

`--max-gap 0.15` 控制线性/旋转插值最大跨度；`--uwb-max-age 0.3` 单独控制距离有效期。
UWB 原本约几 Hz，放到 50 Hz 网格不代表得到 50 Hz 新测量。两端 UWB 分列保留，
不平均、不把两端回调当成同一次硬件测量。方向存在时保留为 `uwb_direction_local`，
不擅自用 IMU 校准旋转它；其本地坐标语义需要另行确认。
UWB 的 app 回调时间也不等同于已知的硬件测量时刻。

## 输出与读取

- `aligned_sensors.npz`：空间校准前的共同时间轴数据；校准失败时仍可用。
- `aligned_calibrated.npz`：上述字段 + 空间校准后的加速度、姿态、角速度和固定参数。
- `alignment.json`：时钟模型、FPS、模态缺失、滤波和有效帧统计。
- `native_modalities.npz`：原始采样率/时间戳/序号的所选模态副本，保留所有原始 UWB 样本。
  注意其中 motion 向量/矩阵已经过仓库坐标约定转换；不是 NDJSON 的逐字节副本。
- `calibration.json`、`calibration_quality.json`、`diagnostic.png`：校准参数、检查和图。
- `iphone-calibrated.npz` / `apple_watch-calibrated.npz`：校准结束后的原采样率补充输出。

设 `T` 为共同时间点数，`S` 为所选设备数（默认 2）：

| 主字段 | 形状 | 含义 |
| --- | --- | --- |
| `timestamp_unix_ns` | `[T] int64` | 同一参考时间轴的网格纳秒时间戳；默认每 20,000,000 ns 一帧 |
| `time_relative_s` | `[T]` | 相对于网格首帧的秒数 |
| `source_names/source_slots` | `[S]` | 设备列名与身体槽位 |
| `source_placements/source_target_frames` | `[S]` 字符串 | 位置标签 / 目标坐标（手机手持为 wrist proxy），与设备列顺序一致 |
| `placement_selection` | 字符串标量 | `explicit` / `legacy_defaults_assumed` / `numeric_slots_unspecified` |
| `legacy_slot_export_available` | bool 标量 | 是否槽位唯一、可无冲突导出旧 7 槽格式；不是模型适配声明 |
| `user_acc_S_m_s2` | `[T,S,3]` | 仓库 S 轴中的去重力加速度，m/s² |
| `gyro_S_rad_s` | `[T,S,3]` | 仓库 S 轴中的融合角速度，rad/s |
| `RIS/RSI` | `[T,S,3,3]` | 每帧传感器↔惯性旋转 |
| `motion_valid` | `[T,S] bool` | 该设备 motion 数据覆盖且插值有效 |
| `joint_motion_valid` | `[T] bool` | 所有设备 motion 都有效，**不含 UWB/raw 流条件** |
| `motion_bracket_unix_ns` | `[T,S,2] int64` | 插值前后原始样本的未校正时间戳 |
| `motion_bracket_sequence` | `[T,S,2] int64` | 原始样本序号；低通可用到更宽邻域，滤波设置保存在报告 |
| `motion_bracket_span_s` | `[T,S]` | 插值两端样本在共同时间轴上的间隔 |
| `motion_exact_sample` | `[T,S] bool` | 网格时刻与原始采样时刻相同；默认滤波后数值仍可能变化 |
| `raw_acc_apple_g` | `[T,S,3]` | 含重力 raw 加速度，原 Apple 轴，单位 g |
| `raw_gyro_apple_rad_s` | `[T,S,3]` | 独立 raw 角速度，原 Apple 轴，rad/s |
| `raw_mag_apple_uT` | `[T,S,3]` | 独立 raw 磁力计，原 Apple 轴，µT |
| 上述 raw 字段 + `_valid` | `[T,S] bool` | 对应流的有效性，不能用 motion_valid 代替 |
| 上述 raw 字段 + `_bracket_unix_ns` | `[T,S,2] int64` | 对应模态的原始插值样本时间 |
| `uwb_distance_m` | `[T,S]` | 对应设备报告的测距，m；无效为 NaN |
| `uwb_valid/uwb_updated` | `[T,S] bool` | 未过期有效 / 本网格点选中了不同于上一点的测量 |
| `uwb_age_s` | `[T,S]` | 当前网格时刻减去被选测量的校正时间 |
| `uwb_measurement_unix_ns` | `[T,S] int64` | 被选测量的原始时间；没有历史样本时 -1 |
| `uwb_measurement_aligned_unix_ns` | `[T,S] int64` | 同一测量映射到共同时钟后的时间 |
| `uwb_sequence_number` | `[T,S] int64` | 被选测量序号；没有历史样本时 -1 |
| `uwb_direction_local/uwb_direction_valid` | `[T,S,3]` / `[T,S]` | 可选 UWB 本地方向及有效性 |
| `RMI/RSB` | `[S,3,3]` | 按设备独立保存的固定参数；JSON schema 2 的权威参数为 `source_calibration[设备]` |
| `acc_M_m_s2/gyro_B_rad_s` | `[T,S,3]` | 模型轴加速度 / 身体部位轴角速度 |
| `ori_MB` | `[T,S,3,3]` | 身体部位→模型旋转 |
| `after_calibration` | `[T] bool` | 最后静止校准窗口结束之后的帧 |

无效网格点仍可能保留用于诊断的 bracket/最近 UWB 样本信息，**必须以 valid mask 为准**。
高频原始 UWB 多个样本可能落入同一个网格间隔；网格只持有最新一个，完整样本见 native archive。
同侧戴表并持手机时 `source_slots` 可以重复，**不能按槽位覆盖/合并设备列**。
这时 JSON 顶层旧 7 槽 `RMI/RSB` 为 `null`，而 `source_calibration` 与 NPZ 中的
`[S,3,3]` 参数完整可用。槽位不重复时仍额外导出旧 7 槽矩阵供兼容读取。
位置元信息同时保存在两个 aligned NPZ、native archive、alignment/calibration/quality JSON 中。

```python
import numpy as np
x = np.load("data/offline_calibration/session_001/aligned_calibrated.npz", allow_pickle=False)
phone = x["source_names"].tolist().index("iphone")
watch = x["source_names"].tolist().index("apple_watch")
print(x["source_placements"].tolist(), x["source_target_frames"].tolist())
keep = x["after_calibration"] & x["joint_motion_valid"]
times = x["timestamp_unix_ns"][keep]
acc = x["acc_M_m_s2"][keep]        # [T_valid, 2, 3]
ori = x["ori_MB"][keep]           # [T_valid, 2, 3, 3]
uwb = x["uwb_distance_m"][keep]   # 仍需结合 uwb_valid
uwb_valid = x["uwb_valid"][keep]
```

原始样本副本按 `设备__模态__字段` 命名，例如
`iphone__accelerometer__timestamp_unix_ns`、`apple_watch__uwb_ranging__values`；
对应 `value_names` 指明列名。motion 副本为 `设备__motion__...`。

## 其它模式与限制

```bash
# 没有校准动作，只做多模态对齐
python apple_offline_calibration.py /path/to/session --alignment-only \
  --fps 50 --output data/offline_calibration/alignment_only

# 用原生 IMU 样本计算校准参数，同时依然输出对齐数据
python apple_offline_calibration.py /path/to/session --calibration-sampling native \
  --output data/offline_calibration/native_calibration
```

手动窗口和历史回放参见 [校准说明](APPLE_OFFLINE_CALIBRATION.md)。历史回放明确不进入新对齐流程，
避免改变复现旧行为的定义。默认输出保留网格全序列，不跨大缺口，不把 NaN 补成零。
`--no-apply` 不输出 calibrated 数组，但保留对齐前处理和校准报告。

目前不对齐 WAV、GPS、PPG 等其它模态，不做人体模型推理。`RSB` 假定设备安装位置不变，
没有解决口袋松动、CoreMotion 重启或长时间姿态漂移。身体动作本身不能证明 N-pose 正确。
缺口/候选歧义会拒绝校准，程序不会利用旧 RMI/RSB 调整时间以“凑近”历史结果。

## 本地验证

继续使用之前两条中的 `seq_03_154502`，没有扩大真实序列跑批：
共同 50 Hz 网格 8,711 帧，两端 motion 全部有效；手表/手机 UWB 原始 890/878 条，
在 0.3 秒有效期下有 7,638/7,529 个有效网格点。旧手表文件没有 raw gyro/magnetometer，
对应数组保持 NaN + false，融合角速度正常保留。空间校准通过信号检查，
但与历史 Watch 参数的较大差异并没有因此被自动“修好”，时间同步准确度仍未独立验证。

测试：仓库根目录 `python -m unittest discover -s tests -v`。
覆盖合成真值、原在线算法回归、时钟 offset/drift、SLERP 环绕、插值缺口、抗混叠、
UWB 因果保持/过期/原始时间、缺失 raw 模态以及分设备文件的完整 CLI 工作流。
