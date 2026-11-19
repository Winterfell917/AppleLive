"""Two-view SMPL MP4 rendering; display transforms never alter saved poses."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

BASE = Path(__file__).resolve().parents[1]


def render_mesh(pose, output, times, bounds, fps, render_config,
                title="Left wrist + right pocket | MobilePoser | fixed root",
                subtitle="Exploratory result: saved RMI is unverified"):
    """Fixed cameras and zero global translation. No smoothing/pose correction."""
    import cv2
    import open3d as o3d
    from articulate.model import ParametricModel

    size = int(render_config["panel_size"])
    yaws = render_config["view_yaws_deg"]
    width, height = size * len(yaws), size + 100
    body = ParametricModel(BASE / "smpl/basicmodel_m.pkl", device=torch.device("cpu"))
    _, zero = body.get_zero_pose_joint_and_vertex()
    center = zero.mean(dim=0).numpy()
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(zero.numpy() - center),
        o3d.utility.Vector3iVector(np.asarray(body.face, np.int32)))
    mesh.paint_uniform_color([.24, .53, .71])
    mesh.compute_vertex_normals()
    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(width=size, height=size, visible=False):
        raise RuntimeError("Could not create OpenGL render window")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    if not writer.isOpened():
        visualizer.destroy_window()
        raise RuntimeError("An H.264 encoder is required for the preview MP4")
    samples, view = [], visualizer.get_view_control()
    try:
        visualizer.add_geometry(mesh)
        option = visualizer.get_render_option()
        option.background_color = np.array([.96, .97, .98])
        option.mesh_show_back_face = True
        # CPU Lambert shading avoids platform-dependent legacy GL light artifacts.
        option.light_on = False
        option.mesh_color_option = o3d.visualization.MeshColorOption.Color
        shade = o3d.visualization.MeshShadeOption
        option.mesh_shade_option = getattr(shade, "SmoothShade", None) or shade.Color
        camera = o3d.camera.PinholeCameraParameters()
        camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(size, size, size * .95, size * .95,
                                                           (size - 1) / 2, (size - 1) / 2)
        extrinsic = np.eye(4)
        # Model +Y is up; camera image +Y is down. Camera looks from +Z.
        extrinsic[:3, :3] = np.diag([1., -1., -1.])
        extrinsic[2, 3] = 2.65
        camera.extrinsic = extrinsic
        if not view.convert_from_pinhole_camera_parameters(camera, allow_arbitrary=True):
            raise RuntimeError("OpenGL renderer rejected the fixed camera")
        view_rotations = [Rotation.from_euler("y", -yaw, degrees=True).as_matrix() for yaw in yaws]
        light = np.array([.3, .5, 1.])
        light /= np.linalg.norm(light)
        pick = set(np.linspace(0, len(pose) - 1, 6).round().astype(int).tolist())
        for start in range(0, len(pose), 32):
            with torch.inference_mode():
                vertices = body.forward_kinematics(pose[start:start + 32], calc_mesh=True)[2].numpy()
            for local, verts in enumerate(vertices):
                index = start + local
                canvas = np.full((height, width, 3), 248, np.uint8)
                for col, rotation in enumerate(view_rotations):
                    # Rotate a display-only copy; model-space predictions stay untouched.
                    mesh.vertices = o3d.utility.Vector3dVector((verts - center) @ rotation.T)
                    mesh.compute_vertex_normals()
                    brightness = .42 + .58 * np.maximum(np.asarray(mesh.vertex_normals) @ light, 0)
                    mesh.vertex_colors = o3d.utility.Vector3dVector(brightness[:, None] * np.array([.30, .60, .78]))
                    visualizer.update_geometry(mesh)
                    visualizer.poll_events()
                    visualizer.update_renderer()
                    rgb = (255 * np.asarray(visualizer.capture_screen_float_buffer(do_render=True))).astype(np.uint8)
                    canvas[64:64 + size, col * size:(col + 1) * size] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(canvas, "Front" if col == 0 else f"Angled ({yaws[col]} deg)",
                                (col * size + 20, 90), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 80, 80), 1, cv2.LINE_AA)
                cv2.putText(canvas, title, (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .63, (40, 40, 40), 1, cv2.LINE_AA)
                cv2.putText(canvas, subtitle, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .52, (30, 85, 145), 1, cv2.LINE_AA)
                phase = "N-pose fitting window" if bounds[0] <= times[index] < bounds[1] else (
                    "Initial handling / settling" if times[index] < bounds[0] else "Motion evaluation")
                text = f"Recording {times[index]:5.2f}s | {phase} | frame {index + 1}/{len(pose)}"
                cv2.putText(canvas, text, (20, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .52, (55, 55, 55), 1, cv2.LINE_AA)
                writer.write(canvas)
                if index in pick:
                    samples.append(cv2.resize(canvas, (640, 370)))
                if (index + 1) % 100 == 0 or index + 1 == len(pose):
                    print(f"Rendering {index + 1}/{len(pose)}", flush=True)
    finally:
        writer.release()
        visualizer.destroy_window()
    if len(samples) == 6:
        cv2.imwrite(str(output.with_name("preview_contact_sheet.jpg")), np.vstack([
            np.hstack(samples[i:i + 2]) for i in range(0, 6, 2)]))
    capture = cv2.VideoCapture(str(output))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != len(pose) or frame_count != len(pose) or abs(actual_fps - fps) > .01:
        raise RuntimeError(f"MP4 validation failed: expected {len(pose)}, decoded {decoded}")
    return {"codec": "H.264", "frames": decoded, "fps": actual_fps, "duration_s": decoded / actual_fps,
            "width": width, "height": height, "translation": "fixed_zero", "post_smoothing": False}
