# Sensor Read Apple IMU live demo

This integration consumes the UDP JSON protocol produced by the sibling
`sensor_read` iOS/watchOS project and sends Apple Watch, iPhone, and AirPods
motion data into MobilePoser.

Default device mapping:

| Sensor Read source | MobilePoser location | Slot |
| --- | --- | ---: |
| `apple_watch` | left wrist | 0 |
| `iphone` | right pocket/thigh | 3 |
| `airpods` | head | 4 |

## Run

1. Put the Mac and iPhone on the same LAN.
2. In Sensor Read, set the destination to the Mac's LAN IP and UDP port 9000.
3. Wear the Watch on the left wrist, put the iPhone in the right pocket, and
   connect compatible AirPods.
4. From the `mobileposer` directory, activate the `mobileposer` Conda env.
5. Start Sensor Read collection, then run:

```bash
python livedemo_apple.py --viewer none --duration 20
```

This first command validates reception, calibration, and inference without a
viewer. The normal command automatically opens a lightweight local stick-figure
window after calibration:

```bash
python livedemo_apple.py
```

The live demo runs inference at 25 FPS by default; override this with `--fps`
when needed. The skeleton preview refreshes at most 30 FPS while inference
retains its configured frame rate. Rendering runs in a separate process, its one-frame queue
drops an old display frame whenever necessary, and that display process computes
only 24 SMPL joint positions—never mesh vertices. Change its refresh cap with
`--viewer-fps 10`.
The 640×640 window opens centered on the primary display and its title bar
reports the actual render FPS once per second. Override its size with, for
example, `--viewer-width 720 --viewer-height 720`. Closing the preview window
does not stop mocap. The
previous Unity viewer remains available with `--viewer unity`; start the Unity
MotionViewer client before using that option.

The default `auto` device selects CPU on Apple Silicon because this project's
small packed-LSTM inference is much faster on CPU than on its current MPS
runtime.  MPS can still be tested explicitly with `--device mps`.

After initial clock alignment, live inference independently takes the latest
received sample from every configured device and concatenates those values for
the network. Device timestamps retain their frozen clock offsets, but live
frames are not required to share one watermark. A sample older than
`--stale-after` is reported through `valid=False` while its latest acceleration,
angular velocity, and rotation remain available to inference.

The same native-frame matching is applied during calibration. The program marks
the physical N-pose or walking window, waits only until every source has
actually delivered data through the end of that window, uses the sparsest
stream (normally Watch) as the time anchor, and pairs nearest real samples from
the other streams. Calibration fails explicitly if a source cannot cover the
window.

The physical calibration windows remain fixed at two seconds for N-pose and
three seconds for the forward step. After each window, the receiver allows up
to five additional seconds for delayed WatchConnectivity frames to arrive; this
does not extend the motion being calibrated. Once a source has delivered a real
frame through the window end, temporary stream staleness no longer invalidates
that completed window. Change only the arrival grace with
`--calibration-wait-timeout 8`. A timeout reports how far each missing stream is
behind and how long ago its latest packet reached the Mac.

Useful variants:

```bash
# More robust yaw/reference alignment using a forward step
python livedemo_apple.py --calibration walking_6dof

# Run without a learned calibrator
python livedemo_apple.py --calibrator nocalibration

# Apply the CHI 2027 cross-device calibration model
python livedemo_apple.py --calibrator ours

# Run independent no-calibration and ours branches on every frame
python livedemo_apple.py --compare-ours-nocalibration

# Apply the CHI 2027 causal Plain Transformer baseline
python livedemo_apple.py --calibrator plain

# Listen to a subset or change body slots (repeat once per source)
python livedemo_apple.py --source-slot apple_watch:0 --source-slot iphone:3
```

## Visualize one sensor modality

Use `--visualize-sensor` and `--visualize-modality` together to open a rolling
three-axis plot while mocap inference continues. Available sensors are
`apple_watch`, `iphone`, and `airpods`. Available modalities are
`acceleration` (sensor-frame `aS`), `aI` (inertial-frame acceleration),
`angular_velocity`, and `orientation`; orientation is displayed as roll, pitch,
and yaw in degrees.

```bash
# iPhone user acceleration, showing the latest 10 seconds
python livedemo_apple.py --no-viewer \
  --visualize-sensor iphone \
  --visualize-modality acceleration

# Apple Watch angular velocity, showing the latest 20 seconds
python livedemo_apple.py --no-viewer \
  --visualize-sensor apple_watch \
  --visualize-modality angular_velocity \
  --visualize-window 20

# AirPods orientation as roll/pitch/yaw
python livedemo_apple.py --no-viewer \
  --visualize-sensor airpods \
  --visualize-modality orientation

# iPhone gravity-free acceleration transformed from S into I
python livedemo_apple.py --no-viewer --calibration none \
  --source-slot iphone:3 \
  --visualize-sensor iphone \
  --visualize-modality aI
```

The default walking calibration aligns each device's independently initialized
CoreMotion yaw frame and converts Apple's Z-up reference into MobilePoser's
Y-up frame.  `npose` is available as a quick integration check when all streams
already share a common reference.  The Apple app sends gravity-free
`userAcceleration` in g; the adapter converts it to m/s² and rotates it from
device coordinates into the calibrated MobilePoser world frame. Walking
calibration asks you to stand in the N-pose and press Enter once. Keep still
while the static N-pose calibration runs; when the computer beeps, step forward
immediately. The beep opens a fixed three-second integration window, after
which the forward direction is computed and mocap starts. It does not use a
countdown or wait for step detection. No static accelerometer bias is estimated
or subtracted. On Windows the cue is a 440 Hz native tone lasting 600 ms. On
hosts without `winsound`, it falls back to the terminal bell.

## Coordinate and variable convention

The Apple receiver uses the same coordinate naming convention as the Huawei
receiver: `M` is the model/world frame, `I` the inertial frame, `S` the sensor
frame, and `B` the bone frame. For any coordinate frames `X` and `Y`, `RXY`
maps a vector from `Y` into `X`. Therefore:

```text
aI  = RIS · aS
aM  = RMI · aI
RMB = RMI · RIS · RSB
```

`aS` is Sensor Read's gravity-free user acceleration, `RIS` is the CoreMotion
sensor-to-inertial orientation, `RMI` is estimated by walking calibration, and
`RSB` is the bone-to-sensor mounting rotation estimated from the N-pose. Saved
calibration metadata uses the keys `RMI` and `RSB`; recorded `acc`, `ori`, and
`gyro` tensors contain `aM`, `RMB`, and `gyroS`, respectively.

Based on device-axis tests, incoming Apple S/I data is treated as left-handed.
The receiver applies the Z reflection `C = diag(1, 1, -1)` before calibration:

```text
aS_right   = C · aS_left
RIS_right  = C · RIS_left · C
aI_right   = RIS_right · aS_right = C · aI_left
gyroS_right = det(C) · C · gyroS_left
```

Consequently, the `acceleration` and `aI` plots show the converted right-handed
values. Because the original CoreMotion inertial frame was +Z-up, physical up
is `-Z` after this reflection; walking calibration uses `(0, 0, -1)` as its
inertial up vector.

## Curated dataset sequences

Use sequence mode to turn one Sensor Read collection into one curated dataset
sample. Recording starts immediately after calibration; the
Sensor Read **End Collection** button marks the end boundary and authorizes the
save. No fixed duration is required.

```bash
python livedemo_apple.py --no-viewer \
  --calibration walking_6dof \
  --subject s01 \
  --name walk_001 \
  --action walk \
  --trial 001
```

After calibration, recording starts immediately without another cue; the only
beep is the start of the forward-step calibration window. Frames whose latest
device timestamps still precede the recording boundary are discarded. When the
action is complete, tap **End Collection** on Apple Watch. The phone records the Watch
`control_event`; livedemo authorizes saving only when `action_stop=1`,
`accepted=1`, `phone_was_recording=1`, and the `sessionID` matches the active
streams. The iPhone also writes one `recording_end` marker locally and repeats
it five times over UDP as a loss-tolerant fallback. `Ctrl-C`, an exception, a
silent network timeout, a rejected stop, or an event from another `sessionID`
discards the sequence and creates no dataset package.

The initial sequence package is written to
`data/datasets/apple/<subject>/<name>/` and contains:

- `manifest.json`: label, subject, trial, Sensor Read session IDs, exact Unix
  time range, model and coordinate conventions.
- `calibration.json`: calibration method/windows, clock offsets, source-slot
  mapping, and full `RMI`/`RSB` matrices.
- `mocap.pt`: latest-per-device native IMU tensors, freshness masks, and
  inferred SMPL pose at the mocap frame rate.

After Sensor Read has ended and closed its local NDJSON, unlock the paired
iPhone, then pull and crop only the matching session interval:

```bash
python apple_sequence_dataset.py pull \
  --sequence-dir data/datasets/apple/s01/walk_001 \
  --device iPhone
```

The full phone recording is downloaded only into a temporary directory. The
sequence package retains only:

- `raw.ndjson`: original Sensor Read events matching both the `sessionID` and
  selected time interval (250 ms boundary padding by default).
- `alignment.pt`: every native-rate modality plus its nearest-sample mapping
  onto each mocap timestamp; raw source timestamps and clock-corrected
  timestamps are both retained.
- `quality.json`: per-stream sample counts/rates, maximum timestamp gaps,
  sequence-number gaps, and mocap alignment offsets.

For a file exported manually with ShareLink/AirDrop, use the same processing
without device access:

```bash
python apple_sequence_dataset.py prepare \
  --sequence-dir data/datasets/apple/s01/walk_001 \
  --raw-file /path/to/sensor-session.ndjson
```

Multiple sequence packages may reference different intervals of the same long
iPhone recording. The pull step locates the device file from the session-ID
prefix embedded in its filename, and it refuses ambiguous or missing matches.
This removes iPhone-to-Mac UDP loss from the dataset path. Watch samples still
first traverse WatchConnectivity before being written into the iPhone file;
strictly lossless Watch capture would additionally require Watch-local storage.

## 30 FPS dataset postprocessing

`apple_postprocess.py` turns the device-local raw recording and the computer's
`calibration.json` into one frame-aligned 30 FPS sample. For jump-based clock
alignment, make one clear upward jump shortly after starting collection on the
Watch. By default the script searches from 0.2 to 8.2 seconds after the accepted
Watch start event and aligns the acceleration-magnitude peak of Watch, iPhone,
and AirPods to the iPhone peak.

Pull the complete matching NDJSON from the paired iPhone and process it directly:

```bash
python apple_postprocess.py \
  --sequence-dir data/datasets/apple/s01/walk_001 \
  --pull \
  --device-name iPhone
```

Or process a complete recording exported with ShareLink/AirDrop:

```bash
python apple_postprocess.py \
  --sequence-dir data/datasets/apple/s01/walk_001 \
  --raw-file /path/to/full-sensor-session.ndjson
```

If an earlier capture was accidentally saved in ordinary livedemo mode instead
of `--name`, bootstrap its sequence metadata before postprocessing:

```bash
python apple_sequence_dataset.py import-record \
  --sequence-dir data/datasets/apple/s01/test_001 \
  --record-file data/records/apple/apple_mocap_YYYYMMDD_HHMMSS.pt \
  --sequence-name test_001 \
  --action test \
  --trial 001
```

This fallback reconstructs the action boundary from the first and last valid
livedemo IMU timestamps. A future curated capture should still use
`livedemo_apple.py --subject ... --name ...`, which records the explicit Enter/start
and Watch/stop boundaries.

The postprocessor uses the saved per-device clock offsets first, then estimates
the remaining inter-device shift from the jump. It constructs a strict common
30 FPS grid inside both the selected action interval and all three streams'
coverage. Vectors are linearly resampled and rotations use SO(3) spherical
interpolation. A frame is rejected if either neighboring native sample is more
than `--max-gap 0.12` seconds away; this prevents a packet-loss hole from being
silently hidden.

For archival sequences that must retain one common time axis despite device
dropouts, use `--gap-policy mask`. Any output frame bracketed by a native gap
larger than `--max-gap` is encoded with zero sensor values and `valid=False` for
that device, matching the model's missing-IMU convention. The quality report
records the masked frame count and fraction. The default remains `error`.

After left-to-right-handed conversion it saves `aS`, `aI`, `aM`, `RIS`, and
`RMB` for all seven model slots. Unused slots remain zero, matching model
training. The first five slots form the same 60-value model input used by
`livedemo_apple.py`.

Model inference deliberately follows the live-demo path rather than an offline
model API: the model is reset once at the beginning, then every 30 FPS input is
passed sequentially to `model.forward_frame`. Therefore `pose[i]` is the output
of the same stateful online step used during live capture. The 45-frame window
returns index 40, giving four actual look-ahead frames. The postprocessor repeats
the final IMU frame four times to flush the window and drops the first four
startup outputs. This compensates the live model delay so `aM[i]`, `RMB[i]`, and
`pose[i]` describe the same target timestamp and retain the same total length.
`forward_offline` is not used.

The sequence directory gains:

- `raw.ndjson`: all raw events for the matching Sensor Read session, including
  the jump/calibration prelude needed for synchronization.
- `processed_30fps.pt`: identical-length tensors. `aS`, `aI`, and `aM` have
  shape `[T, 7, 3]`; `RIS` and `RMB` are `[T, 7, 3, 3]`; `model_input` is
  `[T, 60]`; predicted `pose` is `[T, 24, 3, 3]`.
- `sync_jump.png`: diagnostic plot of the three detected peaks after alignment.
- `postprocess_quality.json`: shifts, native stream coverage, gap checks, output
  interval, frame count, model, device, and inference mode.
- `preview_30fps.mp4`: SMPL mesh preview with exactly `T` frames at 30 FPS.

Use `--no-video` to skip only MP4 generation. If a sequence has no deliberate
jump, `--sync-mode clock` uses calibration clock offsets alone and is expected
to be less accurate. Jump detection can be tuned with `--jump-search-offset`,
`--jump-search-seconds`, and `--min-jump-prominence`. This stage does not yet
align external ground-truth SMPL data; that can be added later without changing
the processed IMU/pose time axis.

## CHI 2027 direct-rotation calibrators

The `ours` option loads
`data/checkpoints/chi2027_calibrator_ours/best.pt`. It applies causal temporal
and cross-device attention over a rolling 125-frame window, scales acceleration
by 30 as in training, and replaces only slots 0/3/4 rotations before frozen
MobilePoser inference. `plain` loads the per-frame-device-concatenation baseline
from `data/checkpoints/chi2027_calibrator_plain/best.pt`.

`--compare-ours-nocalibration` maintains two independent MobilePoser recurrent
states. The lightweight skeleton viewer displays the `Calibration` result; the
Unity viewer displays `NoCalibration` and `Calibration` side by side. Recordings retain
`pose` as the compatible primary (`ours`) output and also store
`pose_nocalibration`, `pose_ours`, and `ori_nocalibration`.
