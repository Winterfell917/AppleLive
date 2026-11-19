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

Sensor Read 的默认设备映射为：

| 数据源 | 身体位置 | 模型槽位 |
| --- | --- | ---: |
| Apple Watch | 左腕 | 0 |
| iPhone | 右口袋/右大腿 | 3 |
| AirPods | 头部 | 4 |

项目支持只使用 Apple Watch 和 iPhone。两设备模式使用：

```bash
python livedemo_apple.py \
  --source-slot apple_watch:0 \
  --source-slot iphone:3 \
  --calibrator nocalibration
```

未配置的 AirPods/头部槽位会以零值填入 MobilePoser 的固定五槽输入，因此可以
完成实时推理，但精度和稳定性通常低于 Watch + iPhone + AirPods 三设备配置。
两设备模式只支持 `none`/`nocalibration`；`ours`、`plain`、`tic` 以及
`--compare-ours-nocalibration` 均要求槽位 0、3、4 同时存在。

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

## Offline Apple mocap: static RMI → N-pose RSB → mesh video

This workflow runs entirely offline using Sensor Read recordings from **an Apple
Watch on the left wrist and an iPhone in the right thigh pocket**. It reproduces
the static-calibration experiment that was visually reviewed on November 19,
2026. No live UDP connection, AirPods, Unity, learned calibration model, or model
training is needed. The iOS/watchOS collection app is maintained separately in
[SmartApple_APP](https://github.com/zhanglbthu/SmartApple_APP).

The end-to-end sequence is:

```text
Known physical sensor pose, held still
  → static recording → per-device RMI
Worn devices, opening N-pose, then natural movement
  → mocap recording → per-device RSB from the opening N-pose
  → timestamp alignment at 30 Hz → calibrated acceleration/orientation
  → MobilePoser (left wrist + right pocket) → 24-joint SMPL poses → MP4
```

### 1. Code map and prerequisites

| Code | Responsibility |
| --- | --- |
| [apple_static_rmi.py](mobileposer/apple_static_rmi.py) | Static-RMI command-line tool and gravity/stationarity diagnostics |
| [static_calibration.py](mobileposer/sensor_apple/static_calibration.py) | Shared RMI loading, fixed-pose calibration, quiet-window selection, and N-pose RSB math |
| [apple_static_rmi_pose.json](mobileposer/configs/apple_static_rmi_pose.json) | Explicit, per-device fixed-pose target matrices; historical example, **not a universal Apple convention** |
| [apple_static_mocap.py](mobileposer/apple_static_mocap.py) | End-to-end offline RSB fitting, input assembly, inference, and artifact export |
| [apple_static_lw_rp.json](mobileposer/configs/apple_static_lw_rp.json) | Two-device placements, model slots, 30 Hz alignment, N-pose thresholds, and render settings |
| [offline.py](mobileposer/sensor_apple/offline.py) / [alignment.py](mobileposer/sensor_apple/alignment.py) | NDJSON validation, source timestamps, resampling, and optional auxiliary channels |
| [apple_postprocess.py](mobileposer/apple_postprocess.py) | Existing stateful model inference and four-frame look-ahead compensation |
| [mesh_video.py](mobileposer/sensor_apple/mesh_video.py) | Front/angled SMPL views, frame labels, H.264 video, and decode verification |
| [test_apple_static_rmi.py](tests/test_apple_static_rmi.py) / [test_apple_static_mocap.py](tests/test_apple_static_mocap.py) | Calibration, validation guards, slot mapping, and CLI regression tests |

Use the installation instructions above and the `mobileposer` conda environment.
**All commands in this section run from the repository root**, not from inside
`mobileposer/`. Inference uses CPU with two PyTorch threads. The existing assets
must be available at:

```text
mobileposer/data/checkpoints/base_model_12combo.pth
mobileposer/smpl/basicmodel_m.pkl
```

Use SMPL and pretrained assets under their applicable licenses. Video rendering
needs a working OpenGL desktop context, Open3D, and an OpenCV build with an H.264
(`avc1`) encoder. An invisible render window is not the same as display-server-free
rendering. Use `--skip-video` for numeric inference on a machine without these
rendering capabilities; do not change the sensor/model pipeline to fix an encoder
or display error.

### 2. Coordinates and what RMI/RSB mean

We use column vectors. **RXY maps coordinates from frame Y into frame X.**

| Frame/matrix | Meaning |
| --- | --- |
| M | Chosen right-handed model/world frame: +Y up; +Z chosen forward; +X completes the frame |
| I | Each device's inertial reference frame; the phone and Watch need not share the same I |
| S | Sensor frame after the existing receiver's coordinate conversion |
| B | Model bone frame associated with the sensor's body slot |
| RIS / RSI | S → I rotation / its transpose I → S |
| RMI | I → M; estimated separately for each device from a known physical orientation |
| RSB | B → S; mounting offset estimated separately from the worn N-pose |
| RMB | B → M; the calibrated bone orientation supplied to the network |

The current repository convention is preserved, including its historical Z-axis
reflection. It is an implementation convention, **not a claim that Apple's
documented axes are left-handed**. With `Q` the matrix stored as `rotation_m11`
through `rotation_m33` in NDJSON:

```python
C = diag(1, 1, -1)
RIS = C @ Q.T @ C
RSI = RIS.T
aS = C @ user_accel_g * 9.80665       # gravity-removed acceleration, m/s²
gyroS = -C @ rotation_rate_rad_s      # axial-vector reflection convention
```

Do not apply this conversion twice. Raw JSON is never rewritten. The model uses
`device_motion` user acceleration and attitude, not the independent raw
accelerometer channel that includes gravity.

### 3. Record and calculate RMI from a known static pose

1. Define model +Y (up) and +Z (your chosen forward direction) physically.
2. Place each device in a **known orientation**, with its axes identified. The
   devices can have different orientations; specify a separate target for each.
3. Start a static recording, allow handling to settle, hold the known poses for
   at least 5–10 seconds, and stop. Recording 12–20 seconds leaves useful margins.
4. Wait for the Watch's files to arrive on the iPhone, then copy the complete
   session folder to the computer. Both `iphone-events-*.ndjson` and
   `apple_watch-events-*.ndjson` must be present and have the same `sessionID`.
   `.wav`, receipt, and session-info files may remain in the folder; these tools
   only read the immediate `*.ndjson` files and do not traverse older sessions.
5. Specify the known rotation `RMS_target` (S → M) in a pose-config JSON. Its
   **columns are the processed sensor's +X, +Y, +Z axes expressed in M**. They must
   be orthonormal with determinant +1. If `P_raw` expresses the independently
   established raw axes in M, use `RMS_target = P_raw @ C` for this pipeline.
   Do not invent a sign change merely to force the determinant to +1.

A stationary sequence does **not** by itself reveal the physical heading. The
known pose supplies information that gravity alone cannot provide. Given the
SO(3) mean of the stable orientations, calculated with SVD projection:

```python
RMI = RMS_target @ mean_RIS_static.T
```

The provided pose config reproduces the earlier back-up sketch experiment:

```text
RMS_target = [-1  0  0
              0  0 -1
              0 -1  0]      # historical example for both devices
```

**That historical target failed the independent gravity-down check on the real
recording.** Keep it unchanged to reproduce that experiment; do not treat it as a
validated mounting recipe for other devices. For a new validated setup, copy the
config and replace `RMS_targets` with independently verified axes. The tool
requires `--pose-config` explicitly rather than silently selecting the example.

```bash
conda activate mobileposer

python mobileposer/apple_static_rmi.py /path/to/static_session \
  --pose-config mobileposer/configs/apple_static_rmi_pose.json \
  --window-seconds 8 \
  --output mobileposer/data/static_rmi/my_static_pose
```

Automatic selection excludes one second at each end of the common coverage and
searches an eight-second window in 0.25-second steps. `--window-seconds` accepts
5–10 seconds. To select a specific interval instead, add `--window START END`,
also spanning 5–10 seconds. All bounds are relative to the **earliest
device-motion sample**, not the button-press time.

Checks include rotation validity, one session per input, increasing timestamps
and sequence numbers, clock discontinuities, a constant attitude-reference-frame
label, and sufficient shared coverage. Static-window limits are:

| Check | Limit |
| --- | ---: |
| Orientation deviation from the mean, P95 | ≤ 2° |
| Angular-speed norm, P95 | ≤ 0.05 rad/s |
| User-acceleration norm, P95 | ≤ 0.05 g |
| Maximum sample gap | ≤ 0.1 s |
| Window coverage | ≥ 95% |
| Gravity angle from model −Y, P95 | ≤ 5° |

Outputs are `calibration.json`, `candidate_rmi.npz`, `diagnostic.png`, and
`pose_config_snapshot.json`. Each source has its own `candidate_RMI`, statistics,
reference-frame counts, and validation issues. Input-file hashes are recorded.

An exit code of **2** can mean the diagnostics were successfully saved but the
coordinate check failed; inspect `calibration.json` rather than assuming no file
was produced. Passing internal checks still does not verify absolute heading or
the correctness of the manually declared physical pose. Existing output folders
are never overwritten.

### 4. Record the mocap sequence and estimate RSB

Wear the Watch on the **left wrist** and put the iPhone in the **right thigh
pocket**. Keep the strap, pocket, and phone orientation fixed throughout the
sequence. After pressing Start, stand in an N-pose facing the chosen model +Z:
upright, arms down naturally, wrists neutral. Allow button interaction to finish,
hold still for several seconds, then perform the movements you want to evaluate.
Moving the phone to another pocket, changing its orientation within the pocket,
or changing the Watch mounting requires a new mounting calibration.

The default config searches only the first **0.5–8.0 seconds**, choosing a common
two-second quiet window in 0.1-second steps. It does not search the whole recording
for a later pause. These worn-pose limits are intentionally less strict than the
tabletop RMI limits: orientation P95 ≤ 2°, gyro P95 ≤ 0.25 rad/s, user acceleration
P95 ≤ 0.08 g, gap ≤ 0.1 s, and coverage ≥ 95%. Stillness is only a quality check;
it cannot identify an N-pose or establish which direction the person faces.

For each device, the N-pose bone target comes from `NPOSE_RMB` in
[sensor.py](mobileposer/sensor_apple/sensor.py):

```text
Left-wrist target (slot 0)       Right-thigh target (slot 3)
[ 0  1  0 ]                    [ 1  0  0 ]
[-1  0  0 ]                    [ 0  1  0 ]
[ 0  0  1 ]                    [ 0  0  1 ]
```

The mounting rotation is fitted without changing the saved RMI:

```python
RSB = (RMI @ mean_RIS_N).T @ target_RMB_N
RMB = RMI @ RIS @ RSB
aI = RIS @ aS
aM = RMI @ aI
aB = RSB.T @ aS                # optional bone-frame acceleration export
```

`RSB` belongs in the orientation chain. **Do not additionally multiply `aM` by
RSB**: `aS` is already a vector in S, and `RMI @ RIS` maps it directly into M.
As a consistency check, `RMB @ aB` equals `aM`.

### 5. Align the recordings and run mocap evaluation

For an RMI artifact that passes the loader's same-session/internal checks:

```bash
python mobileposer/apple_static_mocap.py \
  --session /path/to/mocap_session \
  --rmi mobileposer/data/static_rmi/my_static_pose/calibration.json \
  --config mobileposer/configs/apple_static_lw_rp.json \
  --output mobileposer/data/static_rmi_eval/my_mocap
```

If static RMI and mocap come from **separate recordings**, the inertial reference
may have changed. By default, reuse is refused even if both sessions have the same
attitude-reference-frame label. For an intentional **exploratory replay**, as in
the visually reviewed experiment, append `--allow-unverified-rmi`:

```bash
python mobileposer/apple_static_mocap.py \
  --session /path/to/mocap_session \
  --rmi /path/to/saved_static_rmi/calibration.json \
  --config mobileposer/configs/apple_static_lw_rp.json \
  --npose 4 6 \
  --allow-unverified-rmi \
  --output mobileposer/data/static_rmi_eval/my_exploratory_replay
```

`--npose 4 6` reproduces the reviewed recording's selected interval. It is **not a
universal window**: omit it for opening-window detection or choose checked bounds
appropriate for another sequence. Manual bounds still undergo motion/coverage
checks. The explicit RMI override permits a diagnostic run; it neither changes
the source artifact nor removes its failed-validation status. It cannot bypass
invalid matrices, an incompatible coordinate convention, missing motion data,
or the lack of a usable N-pose interval.

Processing details:

- **Timing:** subtract integer Unix nanoseconds before converting to relative
  seconds; use the intersection of the devices' coverage. The default assumes
  their recorded Unix clocks agree—no physical clock synchronization is inferred.
- **Resampling:** a 30 Hz common grid; SLERP for orientation; linear interpolation
  and offline zero-phase fourth-order low-pass filtering for high-rate vectors
  (cutoff 12 Hz). No extrapolation or interpolation across gaps over 0.15 seconds.
  If any required aligned motion frame is invalid, inference stops.
- **Auxiliary channels:** raw accelerometer/gyro/magnetometer channels and UWB
  are included when present. Missing channels remain missing. UWB uses causal
  hold with a 0.3-second expiry, not fabricated new measurements. Audio is not
  synchronized or used by this runner. See [alignment details](mobileposer/APPLE_OFFLINE_ALIGNMENT.md).
- **Model inputs:** five fixed slots: left wrist (0), right wrist (1), left thigh
  (2), right thigh (3), head (4). Only 0 and 3 are active. Both acceleration and
  orientation matrices in the other slots are **zeros**, not identity matrices.
  The 60-dimensional input is `[flatten(aM / 30), flatten(RMB)]`: 15 acceleration
  values followed by 45 orientation values. Gyroscope, UWB, audio, and raw
  magnetometer are not direct inputs to this checkpoint.
- **Model:** the existing `base_model_12combo.pth`, with no learned calibrator or
  retraining. This is a five-slot checkpoint used with two active slots, not a
  newly trained two-device model. Configuration changes are local to this runner;
  the live demo's global config file is not edited.
- **Temporal state:** reset once per recording, then use `forward_frame` for
  every input. Compensate the model's four-frame look-ahead by appending four
  copies of the final input and discarding four startup outputs. Startup and
  tail boundary effects still exist; poses are not post-smoothed.
- **Rendering:** generic SMPL body, front and angled views, fixed global root,
  30 FPS. Display-only camera/mesh transforms never alter saved model outputs.
  The MP4 is read back to verify frame count, FPS, and complete decoding.

### 6. Outputs, reuse, and interpretation

| Output in the evaluation folder | Contents |
| --- | --- |
| `calibration.json` | Per-source RMI and fitted RSB, body slots, N-pose bounds/quality, source RMI hash, and override reasons |
| `rmi_source_snapshot.json` | Unmodified content of the input RMI artifact |
| `effective_config.json` | Exact placements, thresholds, checkpoint choice, and renderer settings |
| `native_sensor_arrays.npz` | Native sample times and parsed sensor arrays; motion arrays already use the receiver's axis/unit conversion, so this is **not a byte-for-byte raw recording** |
| `aligned_sensors.npz` | Common timestamps, source-axis RIS/RSI, acceleration/gyro, validity masks, interpolation brackets, and available auxiliary channels |
| `model_io.npz` | `input_60d` `[T,60]`, `pose_local_rotation` `[T,24,3,3]`, `acc_M_slots` `[T,5,3]`, `ori_MB_slots` `[T,5,3,3]`, `acc_B_sources` `[T,2,3]`, timestamps, and active-slot mask |
| `mocap_result.pt` | PyTorch tensors, estimated poses, calibration/config snapshots, source ordering, and look-ahead metadata |
| `npose_diagnostics.png` | Full-sequence acceleration, angular speed, and orientation deviation with the selected interval highlighted |
| `mocap_mesh.mp4` / `preview_contact_sheet.jpg` | Full video and six sampled frames; omitted with `--skip-video` |
| `report.json` | Input/checkpoint/code hashes, runtime versions, alignment checks, warnings, and video decode results |

The source order in the default aligned arrays is **Watch, iPhone**, while model
arrays use the five fixed slots above. Always read `source_names`/`source_slots`
rather than assuming a column order. `time_from_recording_origin_s` in the NPZ
and `time_relative_s` in the PT are relative to the earliest motion sample. In
`aligned_sensors.npz`, `time_relative_s` starts at the first common grid sample.
Use `timestamp_unix_ns` to relate the files precisely.

```python
import numpy as np

result = np.load("mobileposer/data/static_rmi_eval/my_mocap/model_io.npz",
                 allow_pickle=False)
pose = result["pose_local_rotation"]  # SMPL local joint rotations, not vertices
acc = result["acc_M_slots"]
orientation = result["ori_MB_slots"]
```

The recorded NDJSON files stay unchanged. Keep them with the calibration and
report for later reprocessing. Prefer the pickle-free NPZ format when sharing;
only load `.pt` files from trusted sources. RSB can be reused only while mounting
and the relevant coordinate references remain unchanged. A new pose/config or
recording should use a new output directory.

The reviewed real-device test selected **4.0–6.0 seconds**, produced **545 frames
at 30 Hz (~18.2 seconds)**, and met the user's visual expectations. Inspect motion
after the N-pose, including left/right correspondence, limb rotation direction,
jitter, implausible bending, and whether motions match what was performed.
Initial button handling is included in the video and should not be mistaken for
the calibration pose. The fixed-root video does not evaluate global translation,
distance traveled, or foot sliding caused by omitting translation.

**Visual plausibility is not quantitative calibration validation.** The N-pose
fit enforces pose closure by construction, and a neural model can hide calibration
errors. In the historical run, the saved sketch-based RMI still had a gravity-sign
validation failure, and cross-session inertial-frame consistency was unverified.
Gravity checks do not resolve yaw; a shared timestamp grid does not prove clock
synchronization. Use independent known poses, repeatability tests, and ultimately
reference motion capture/video before making accuracy claims.

Run the regression suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Raw recordings, generated arrays, personal calibration matrices, and videos belong
under the Git-ignored `mobileposer/data/` tree. They are not included in the source
push. Only source code, generic configs, documentation, and synthetic-data tests
are part of this workflow's Git changes.

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
