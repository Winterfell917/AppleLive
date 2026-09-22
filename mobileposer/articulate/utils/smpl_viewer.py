"""Small out-of-process Pygame viewer for one live SMPL skeleton."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import queue
import time
from typing import Optional

import numpy as np
import torch


DEFAULT_VIEWER_WIDTH = 640
DEFAULT_VIEWER_HEIGHT = 640


def _projected_model_y(joints: np.ndarray) -> np.ndarray:
    return joints[:, 1] + 0.06 * joints[:, 2]


def _project_joints(
    joints: np.ndarray,
    width: int,
    height: int,
    vertical_center: Optional[float] = None,
) -> np.ndarray:
    """Project model-space SMPL joints into a stable, centered 2-D view."""
    # The SMPL pelvis is at y=0 while the feet are near y=-1, so centering on
    # model-space zero pushes the legs below the window.  Center the zero-pose
    # bounds instead, and keep that center fixed while live poses are rendered.
    scale = min(width, height) / 2.70
    root = joints[0]
    centered = joints - np.asarray([root[0], 0.0, root[2]], dtype=np.float32)
    x = centered[:, 0] + 0.22 * centered[:, 2]
    y = _projected_model_y(centered)
    if vertical_center is None:
        vertical_center = float((y.min() + y.max()) * 0.5)
    return np.stack(
        (width * 0.5 + scale * x, height * 0.5 - scale * (y - vertical_center)), axis=1
    ).round().astype(np.int32)


def _center_window(pygame, width: int, height: int) -> None:
    """Set the SDL position before creating a window on the primary display."""
    try:
        desktop_width, desktop_height = pygame.display.get_desktop_sizes()[0]
    except (AttributeError, IndexError, pygame.error):
        os.environ["SDL_VIDEO_CENTERED"] = "1"
        return
    left = max(0, (desktop_width - width) // 2)
    top = max(0, (desktop_height - height) // 2)
    os.environ.pop("SDL_VIDEO_CENTERED", None)
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{left},{top}"


def _viewer_process(frame_queue, status_queue, model_path, title, width, height) -> None:
    """Own the window and joint-only FK without ever blocking inference."""
    try:
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        import pygame
        from articulate.model import ParametricModel

        pygame.init()
        _center_window(pygame, width, height)
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        bodymodel = ParametricModel(model_path, device=torch.device("cpu"))
        parents = tuple(bodymodel.parent)
        zero_joints, _ = bodymodel.get_zero_pose_joint_and_vertex()
        latest_joints = zero_joints.numpy()
        zero_y = _projected_model_y(latest_joints)
        vertical_center = float((zero_y.min() + zero_y.max()) * 0.5)
        scale = min(width, height) / 2.70
        floor_y = min(
            height - 16,
            round(height * 0.5 - scale * (float(zero_y.min()) - vertical_center) + 12),
        )
        status_queue.put(("ready", None))

        background = (246, 248, 251)
        bone_color = (38, 103, 168)
        joint_color = (16, 65, 115)
        text_color = (35, 42, 50)
        floor_color = (205, 211, 219)
        font = pygame.font.SysFont("Arial", 22)
        clock = pygame.time.Clock()
        fps_started_s = time.monotonic()
        rendered_frames = 0
        render_fps = 0.0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pose = None
            try:
                item = frame_queue.get(timeout=0.005)
                if item is None:
                    break
                pose = item
                # Keep only the newest pose; display latency never accumulates.
                while True:
                    try:
                        item = frame_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        running = False
                        break
                    pose = item
            except queue.Empty:
                pass

            if pose is not None and running:
                pose_tensor = torch.from_numpy(pose).reshape(1, 24, 3, 3)
                with torch.inference_mode():
                    latest_joints = bodymodel.forward_kinematics(
                        pose_tensor, calc_mesh=False
                    )[1][0].numpy()
                rendered_frames += 1

            screen.fill(background)
            points = _project_joints(latest_joints, width, height, vertical_center)
            pygame.draw.line(screen, floor_color, (width // 5, floor_y), (width * 4 // 5, floor_y), 2)
            for joint, parent in enumerate(parents):
                if parent is None:
                    continue
                pygame.draw.line(
                    screen, bone_color,
                    tuple(points[parent]), tuple(points[joint]), 6,
                )
            for joint, point in enumerate(points):
                radius = 11 if joint == 15 else 6
                pygame.draw.circle(screen, joint_color, tuple(point), radius)

            fps_now_s = time.monotonic()
            fps_elapsed_s = fps_now_s - fps_started_s
            if fps_elapsed_s >= 1.0:
                render_fps = rendered_frames / fps_elapsed_s
                pygame.display.set_caption(f"{title} | Render {render_fps:.1f} FPS")
                fps_started_s = fps_now_s
                rendered_frames = 0
            label = font.render(f"Skeleton preview  {render_fps:.1f} FPS", True, text_color)
            screen.blit(label, (18, 16))
            pygame.display.flip()
            # Twice the default display rate is enough for responsive window
            # events without wasting a CPU core redrawing an unchanged pose.
            clock.tick(60)
    except BaseException as error:
        try:
            status_queue.put(("error", repr(error)))
        except BaseException:
            pass
    finally:
        try:
            import pygame
            pygame.quit()
        except BaseException:
            pass


class LightweightSMPLViewer:
    """Send throttled SMPL poses to a disposable joint-only display process."""

    def __init__(
        self,
        model_path,
        fps: float = 30.0,
        width: int = DEFAULT_VIEWER_WIDTH,
        height: int = DEFAULT_VIEWER_HEIGHT,
        title: str = "Apple Mocap — Skeleton Preview",
    ) -> None:
        if fps <= 0:
            raise ValueError("Viewer FPS must be positive")
        if width <= 0 or height <= 0:
            raise ValueError("Viewer width and height must be positive")
        self.model_path = str(model_path)
        self.frame_period_s = 1.0 / fps
        self.width = width
        self.height = height
        self.title = title
        self._context = mp.get_context("spawn")
        self._frame_queue = None
        self._status_queue = None
        self._process = None
        self._next_render_s = 0.0
        self._open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self) -> None:
        if importlib.util.find_spec("pygame") is None:
            raise RuntimeError(
                "The skeleton viewer requires Pygame. Install it with "
                "`pip install pygame`, or run with --viewer none."
            )
        self._frame_queue = self._context.Queue(maxsize=1)
        self._status_queue = self._context.Queue(maxsize=2)
        self._process = self._context.Process(
            target=_viewer_process,
            args=(self._frame_queue, self._status_queue, self.model_path, self.title, self.width, self.height),
            name="skeleton-preview",
            daemon=True,
        )
        self._process.start()
        try:
            state, detail = self._status_queue.get(timeout=10.0)
        except queue.Empty as error:
            self.close()
            raise RuntimeError("Timed out while opening the skeleton preview window") from error
        if state != "ready":
            self.close()
            raise RuntimeError(f"Could not open the skeleton preview window: {detail}")
        self._open = True
        self._next_render_s = 0.0

    def update(self, pose: torch.Tensor) -> bool:
        """Non-blockingly publish the latest pose when its display frame is due."""
        if not self._open or self._process is None or not self._process.is_alive():
            self._open = False
            return False
        now = time.monotonic()
        if now < self._next_render_s:
            return True
        self._next_render_s = now + self.frame_period_s
        pose_array = pose.detach().cpu().numpy()
        try:
            self._frame_queue.put_nowait(pose_array)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frame_queue.put_nowait(pose_array)
            except queue.Full:
                pass
        return True

    def close(self) -> None:
        if self._frame_queue is not None and self._process is not None:
            try:
                self._frame_queue.put_nowait(None)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(None)
                except queue.Full:
                    pass
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        for channel in (self._frame_queue, self._status_queue):
            if channel is not None:
                channel.close()
                channel.join_thread()
        self._open = False
        self._process = None
        self._frame_queue = None
        self._status_queue = None
