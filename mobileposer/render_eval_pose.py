"""Infer a pose from one collected sequence and save a pyrender MP4.

The mesh, camera, light, and offscreen renderer follow SmartWear
``pos_cls/pos_visualize.py``. Watch and phone stay on the recorded slots:
green at the wrist, cyan at the pocket. There is no slot classifier.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.pop("PYOPENGL_PLATFORM", None)

import cv2
import numpy as np
import torch

from config import amass, model_config, paths
from utils.model_utils import load_model


BASE_DIR = Path(__file__).resolve().parent
SLOT_NAME = {0: "L_wrist", 1: "R_wrist", 2: "L_pocket", 3: "R_pocket"}
SLOT_VERTEX = {0: 1961, 1: 5424, 2: 876, 3: 4362}
WATCH_RGBA = (0.15, 0.80, 0.25, 1.0)
PHONE_RGBA = (0.15, 0.65, 0.95, 1.0)


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = eye - target
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(up, forward)
    right /= np.linalg.norm(right) + 1e-8
    new_up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = new_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def open_writer(path: Path, fps: float, width: int, height: int):
    for fourcc in ("mp4v", "avc1", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if writer.isOpened():
            return writer, fourcc
        writer.release()
    raise RuntimeError(f"Cannot open VideoWriter for {path}")


class MeshViewer:
    """Offscreen SMPL viewer with the SmartWear scene setup."""

    def __init__(self, faces, width: int = 960, height: int = 720) -> None:
        import pyrender
        import trimesh

        self.pyrender = pyrender
        self.trimesh = trimesh
        self.faces = np.asarray(faces, dtype=np.int64)
        self.width = width
        self.height = height
        self.scene = pyrender.Scene(
            bg_color=[30, 30, 36, 255],
            ambient_light=[0.35, 0.35, 0.35],
        )
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width / height)
        self.scene.add(camera, pose=look_at([0.0, 1.15, 3.1], [0.0, 0.75, 0.0]))
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=look_at([1.5, 3.0, 2.0], [0.0, 0.8, 0.0]))
        self.mesh_node = None
        self.marker_nodes = []
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def _sphere(self, xyz, radius, rgba):
        sphere = self.trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(np.asarray(xyz, dtype=np.float64))
        color = (np.array(rgba[:3]) * 255).astype(np.uint8)
        sphere.visual.vertex_colors = np.tile(color, (sphere.vertices.shape[0], 1))
        return self.pyrender.Mesh.from_trimesh(sphere, smooth=True)

    def render_bgr(self, vertices, watch_slot, phone_slot, caption) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float32)
        body = self.trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        body.visual.vertex_colors = [190, 190, 200, 255]
        mesh = self.pyrender.Mesh.from_trimesh(body, smooth=True)
        if self.mesh_node is not None:
            self.scene.remove_node(self.mesh_node)
        self.mesh_node = self.scene.add(mesh)
        for node in self.marker_nodes:
            self.scene.remove_node(node)
        self.marker_nodes = []
        for slot, rgba in ((watch_slot, WATCH_RGBA), (phone_slot, PHONE_RGBA)):
            if slot not in SLOT_VERTEX:
                continue
            self.marker_nodes.append(
                self.scene.add(self._sphere(vertices[SLOT_VERTEX[slot]], 0.048, rgba))
            )
        color, _ = self.renderer.render(self.scene)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        y = 28
        for line in caption:
            cv2.putText(
                bgr, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA,
            )
            y += 24
        return bgr

    def close(self) -> None:
        self.renderer.delete()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_labels(sequence: Path) -> tuple[Path, dict]:
    path = sequence / "labels.pt" if sequence.is_dir() else sequence
    if not path.is_file():
        raise SystemExit(f"找不到 {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path.parent, payload


def signals_from_labels(payload: dict) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Place calibrated aM and RMB into the five-slot MobilePoser input."""
    acceleration_s = payload["aS"].float()
    orientation = payload["RIS"].float()
    rotation_mi = payload["R_MI"].float()
    rotation_sb = payload["R_SB"].float()
    slots = [int(slot) for slot in payload["model_slots"]]
    if len(slots) != acceleration_s.shape[1]:
        raise RuntimeError("model_slots 与设备数量不一致")
    frame_count = acceleration_s.shape[0]
    count = model_config.n_joints
    acceleration_m = torch.zeros(frame_count, count, 3)
    rotation_mb = torch.zeros(frame_count, count, 3, 3)
    for index, slot in enumerate(slots):
        if slot < 0 or slot >= count:
            raise RuntimeError(f"槽位 {slot} 超出模型输入 0..{count - 1}")
        acceleration_i = torch.matmul(
            orientation[:, index], acceleration_s[:, index].unsqueeze(-1)
        ).squeeze(-1)
        acceleration_m[:, slot] = torch.matmul(
            rotation_mi[index], acceleration_i.unsqueeze(-1)
        ).squeeze(-1)
        rotation_mb[:, slot] = rotation_mi[index] @ orientation[:, index] @ rotation_sb[index]
    return acceleration_m, rotation_mb, slots[0], slots[1]


def infer_pose(acceleration_m: torch.Tensor, rotation_mb: torch.Tensor, model_path: Path, device: torch.device) -> torch.Tensor:
    model_config.device = device
    paths.smpl_file = BASE_DIR / "smpl/basicmodel_m.pkl"
    model = load_model(str(model_path)).to(device).eval()
    model.reset()
    model_input = torch.cat(
        (
            (acceleration_m / amass.acc_scale).flatten(1),
            rotation_mb.flatten(1),
        ),
        dim=1,
    ).to(device)
    output_index = int(model.num_past_frames)
    lookahead = int(model.num_total_frames) - output_index - 1
    inference_input = torch.cat(
        (model_input, model_input[-1:].repeat(lookahead, 1)),
        dim=0,
    )
    poses = []
    with torch.inference_mode():
        for index, frame in enumerate(inference_input):
            poses.append(model.forward_frame(frame).view(24, 3, 3).cpu())
            if (index + 1) % 100 == 0 or index + 1 == len(inference_input):
                print(f"\r姿态推理 {index + 1}/{len(inference_input)}", end="", flush=True)
    print()
    return torch.stack(poses)[lookahead:lookahead + len(model_input)]


def render_video(
    pose: torch.Tensor,
    output: Path,
    watch_slot: int,
    phone_slot: int,
    title: str,
    fps: float,
    width: int,
    height: int,
) -> None:
    from articulate.model import ParametricModel

    body = ParametricModel(BASE_DIR / "smpl/basicmodel_m.pkl", device=torch.device("cpu"))
    viewer = MeshViewer(body.face, width=width, height=height)
    writer, fourcc = open_writer(output, fps, width, height)
    frame_count = len(pose)
    try:
        for start in range(0, frame_count, 32):
            with torch.inference_mode():
                vertices = body.forward_kinematics(
                    pose[start:start + 32], calc_mesh=True
                )[2].numpy()
            for offset, frame_vertices in enumerate(vertices):
                index = start + offset
                caption = [
                    f"{title}  t={index / fps:5.2f}s  {index + 1}/{frame_count}",
                    f"watch={SLOT_NAME.get(watch_slot, watch_slot)}  "
                    f"phone={SLOT_NAME.get(phone_slot, phone_slot)}",
                    "watch green | phone cyan",
                ]
                image = viewer.render_bgr(frame_vertices, watch_slot, phone_slot, caption)
                if image.shape[1] != width or image.shape[0] != height:
                    image = cv2.resize(image, (width, height))
                writer.write(np.ascontiguousarray(image))
            print(f"\r渲染 {min(start + 32, frame_count)}/{frame_count}", end="", flush=True)
    finally:
        writer.release()
        viewer.close()
    print()
    print(f"已保存 {output} ({fourcc})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用采集序列离线推理姿态，并按 SmartWear 的 pyrender 保存 mp4")
    parser.add_argument("--sequence", type=Path, required=True, help="序列目录或 labels.pt")
    parser.add_argument("--output", type=Path, default=None, help="mp4 路径，默认写到序列目录的 pose.mp4")
    parser.add_argument("--model", type=Path, default=paths.weights_file)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--fps", type=float, default=None, help="默认使用 labels.pt 里的 fps")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width 和 --height 必须为正")
    sequence_dir, payload = load_labels(args.sequence)
    fps = float(args.fps if args.fps is not None else payload.get("fps", 30.0))
    if fps <= 0:
        raise SystemExit("--fps 必须为正")
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise SystemExit(f"找不到模型 {model_path}")
    output = args.output or (sequence_dir / "pose.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"推理设备 {device}")
    acceleration_m, rotation_mb, watch_slot, phone_slot = signals_from_labels(payload)
    pose = infer_pose(acceleration_m, rotation_mb, model_path, device)
    title = sequence_dir.name
    render_video(pose, output, watch_slot, phone_slot, title, fps, args.width, args.height)


if __name__ == "__main__":
    main()
