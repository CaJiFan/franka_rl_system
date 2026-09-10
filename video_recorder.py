#!/usr/bin/env python3
"""
video_recorder.py
=================
Interactive OpenCV (cv2) video recording application for robot wiping experiments.

Features:
- Live camera preview with on-screen clickable buttons:
  [REC/STOP], [PAUSE], [SAVE], [SNAPSHOT], [CAM: idx], [EP: #], [QUIT].
- Quick camera switching: click [CAM: idx] or press TAB to cycle camera devices at runtime.
- Support for non-Orbbec camera selection (e.g. Creative Senz3D, standard webcams).
- Clean frame recording: saved MP4 file captures the pure camera feed without UI clutter.
- Auto-save on stop: stopping a recording immediately finalizes and saves the video.
- Asynchronous threaded video writing to prevent frame drops at high resolutions.
- Episode and run tagging (e.g. wiping_20260910_153022_ep01.mp4).
- Keyboard shortcuts for hands-on robot operation.
- Standalone CLI execution or importable `VideoRecorder` class.

Usage:
  .venv/bin/python video_recorder.py --camera 10
  .venv/bin/python video_recorder.py --non-orbbec
  .venv/bin/python video_recorder.py --select-camera
  .venv/bin/python video_recorder.py --list-cameras
"""

import os
import sys
import glob
import re
import time
import queue
import threading
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

# Silence verbose OpenCV and Qt warnings
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2
import numpy as np

try:
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def get_camera_name(idx: int) -> str:
    """Reads the hardware device model name from Linux sysfs."""
    p = Path(f"/sys/class/video4linux/video{idx}/name")
    if p.exists():
        raw = p.read_text().strip()
        # Clean up repeated trailing vendor tags like ': Creativ' or ': Or'
        clean = re.sub(r':\s*\w+$', '', raw)
        return clean
    return f"Camera {idx}"


def get_system_video_indices() -> List[int]:
    """Finds all /dev/video* device numbers on Linux systems."""
    dev_paths = glob.glob("/dev/video*")
    indices = []
    for p in dev_paths:
        m = re.search(r"/dev/video(\d+)", p)
        if m:
            indices.append(int(m.group(1)))
    return sorted(list(set(indices)))


def list_available_cameras(primary_only: bool = True) -> List[Dict]:
    """Probes existing video devices and returns working camera details."""
    available = []
    candidates = get_system_video_indices()
    if not candidates:
        candidates = [0, 1, 2, 3, 10]

    seen_models = set()
    for idx in candidates:
        try:
            name = get_camera_name(idx)
            is_orbbec = "orbbec" in name.lower()

            # Group primary video nodes by model name to avoid cluttering secondary depth/IR nodes
            if primary_only and name in seen_models:
                continue

            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Warmup read to verify stream viability
                valid = False
                for _ in range(5):
                    try:
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            valid = True
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                            backend = cap.getBackendName()
                            available.append({
                                "index": idx,
                                "name": name,
                                "is_orbbec": is_orbbec,
                                "width": w,
                                "height": h,
                                "fps": fps,
                                "backend": backend,
                            })
                            seen_models.add(name)
                            break
                    except cv2.error:
                        time.sleep(0.04)
                cap.release()
        except Exception:
            pass
    return available


def prompt_camera_selection(cameras: List[Dict]) -> int:
    """Displays an interactive console menu to choose a camera index."""
    if not cameras:
        print("No camera devices found! Defaulting to index 0.")
        return 0

    print("\n" + "=" * 60)
    print("  SELECT CAMERA DEVICE")
    print("=" * 60)
    for i, cam in enumerate(cameras, 1):
        orbbec_tag = "[Orbbec]" if cam["is_orbbec"] else "[Non-Orbbec]"
        print(f"  [{i}] Index {cam['index']:2d} : {cam['name']} {orbbec_tag}")
        print(f"       Resolution: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS ({cam['backend']})")
    print("=" * 60)

    # Suggest non-orbbec as default if available
    default_choice = 1
    for i, cam in enumerate(cameras, 1):
        if not cam["is_orbbec"]:
            default_choice = i
            break

    while True:
        choice = input(f"Select camera [1-{len(cameras)}, default {default_choice}]: ").strip()
        if not choice:
            return cameras[default_choice - 1]["index"]
        try:
            val = int(choice)
            if 1 <= val <= len(cameras):
                return cameras[val - 1]["index"]
        except ValueError:
            pass
        print(f"Invalid choice '{choice}'. Please enter a number between 1 and {len(cameras)}.")


class AsyncVideoWriter:
    """Threaded video writer to prevent I/O disk writes from stuttering the live preview."""

    def __init__(self, filepath: str, width: int, height: int, fps: float):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.fps = fps
        self.queue: queue.Queue = queue.Queue(maxsize=300)
        self.stop_event = threading.Event()
        self.writer: Optional[cv2.VideoWriter] = None
        self._init_writer()

        self.frames_written = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _init_writer(self):
        # Codec fallback sequence for broad Linux compatibility
        codecs = ["mp4v", "avc1", "XVID", "MJPG"]
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))
            if writer.isOpened():
                self.writer = writer
                return
            writer.release()

        # Fallback to default fourcc 0 if all named codecs failed
        self.writer = cv2.VideoWriter(self.filepath, 0, self.fps, (self.width, self.height))

    def write(self, frame: np.ndarray):
        if not self.stop_event.is_set():
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass  # Avoid memory explosion if disk is stalled

    def _worker(self):
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.writer is not None:
                self.writer.write(frame)
                self.frames_written += 1
            self.queue.task_done()

    def stop(self) -> int:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=3.0)
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        return self.frames_written


class Button:
    """Clickable UI button rendered directly on OpenCV frames."""

    def __init__(self, name: str, label: str, rect: Tuple[int, int, int, int],
                 bg_color: Tuple[int, int, int], text_color: Tuple[int, int, int] = (255, 255, 255)):
        self.name = name
        self.label = label
        self.rect = rect  # (x1, y1, x2, y2)
        self.bg_color = bg_color
        self.text_color = text_color
        self.hovered = False

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw(self, frame: np.ndarray, font_scale: float = 0.46):
        x1, y1, x2, y2 = self.rect
        color = self.bg_color
        if self.hovered:
            color = tuple(min(255, c + 35) for c in self.bg_color)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        border_color = (230, 230, 230) if self.hovered else (80, 80, 80)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 1)

        # Center text inside button
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(self.label, font, font_scale, thickness)
        tx = x1 + (x2 - x1 - text_w) // 2
        ty = y1 + (y2 - y1 + text_h) // 2
        cv2.putText(frame, self.label, (tx, ty), font, font_scale, self.text_color, thickness, cv2.LINE_AA)


class VideoRecorder:
    """Interactive OpenCV Video Recorder with on-screen buttons, camera switching, and auto-save."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720,
                 fps: float = 30.0, output_dir: str = "recordings", prefix: str = "wiping",
                 burn_hud: bool = False, window_name: str = "Wiping Policy Video Recorder"):
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.burn_hud = burn_hud
        self.window_name = window_name

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Camera & Hardware State
        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_width = width
        self.actual_height = height
        self.actual_fps = fps
        self.camera_name = get_camera_name(camera_index)
        self.available_cameras: List[Dict] = []
        self._discover_cameras()

        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.async_writer: Optional[AsyncVideoWriter] = None
        self.current_video_path: Optional[str] = None
        self.recording_start_time = 0.0
        self.pause_start_time = 0.0
        self.total_paused_duration = 0.0
        self.recorded_frames = 0
        self.episode_counter = 1

        # Toast notification system
        self.toast_message = ""
        self.toast_color = (0, 200, 0)
        self.toast_expiry = 0.0

        # UI & Buttons
        self.buttons: List[Button] = []
        self.show_hud = True
        self.mouse_pos = (-1, -1)
        self._request_quit = False

    def _discover_cameras(self):
        """Discovers active cameras for runtime switching."""
        self.available_cameras = list_available_cameras(primary_only=True)
        # Ensure current index is in the list
        current_indices = [c["index"] for c in self.available_cameras]
        if self.camera_index not in current_indices:
            self.available_cameras.insert(0, {
                "index": self.camera_index,
                "name": get_camera_name(self.camera_index),
                "is_orbbec": "orbbec" in get_camera_name(self.camera_index).lower(),
                "width": self.target_width,
                "height": self.target_height,
                "fps": self.target_fps,
                "backend": "V4L2",
            })

    def init_camera(self) -> bool:
        """Initializes or connects to the selected camera device."""
        self.camera_name = get_camera_name(self.camera_index)
        print(f"[RECORDER] Connecting to Camera Index {self.camera_index} ({self.camera_name})...")

        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            print(f"[RECORDER] ERROR: Could not open camera at index {self.camera_index}.")
            return False

        # Apply settings
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Warmup loop: read frames until valid frame arrives
        valid = False
        for _ in range(10):
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    valid = True
                    break
            except cv2.error:
                time.sleep(0.04)

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dev_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if dev_fps > 0:
            self.actual_fps = dev_fps

        print(f"[RECORDER] Connected: {self.camera_name} ({self.actual_width}x{self.actual_height} @ {self.actual_fps:.1f} FPS)")
        self._init_buttons()
        return True

    def switch_camera(self, new_index: int) -> bool:
        """Switches to another camera device index cleanly."""
        if self.camera_index == new_index and self.cap is not None and self.cap.isOpened():
            return True

        if self.is_recording:
            self.stop_recording(auto_saved=True)

        old_index = self.camera_index
        print(f"[RECORDER] Switching from Camera {old_index} to Camera {new_index}...")

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_index = new_index
        if not self.init_camera():
            print(f"[RECORDER] Failed to connect to Camera {new_index}, falling back to {old_index}...")
            self.camera_index = old_index
            self.init_camera()
            self.show_toast(f"Failed to switch to Camera {new_index}", color=(0, 0, 220), duration_sec=3.0)
            return False

        cam_label = f"Cam {self.camera_index}: {self.camera_name}"
        self.show_toast(f"✓ Switched to {cam_label}", color=(0, 220, 180), duration_sec=3.5)
        self._update_button_labels()
        return True

    def cycle_camera(self) -> bool:
        """Cycles to the next available working camera."""
        if not self.available_cameras:
            self._discover_cameras()

        if len(self.available_cameras) <= 1:
            self.show_toast(f"Only 1 camera detected (Cam {self.camera_index})", color=(0, 160, 240), duration_sec=2.0)
            return True

        indices = [c["index"] for c in self.available_cameras]
        if self.camera_index in indices:
            cur_pos = indices.index(self.camera_index)
            next_idx = indices[(cur_pos + 1) % len(indices)]
        else:
            next_idx = indices[0]

        return self.switch_camera(next_idx)

    def _init_buttons(self):
        """Builds bottom toolbar buttons sized proportionally to frame width."""
        w = self.actual_width
        h = self.actual_height

        bar_h = 46 if h >= 600 else 38
        y1 = h - bar_h - 8
        y2 = h - 8

        # 7 buttons: REC, PAUSE, SAVE, SNAPSHOT, CAM SWITCH, EPISODE, QUIT
        btn_specs = [
            ("rec_stop", " [REC] ", (40, 160, 40)),
            ("pause", " PAUSE ", (40, 120, 180)),
            ("save", " SAVE ", (160, 100, 30)),
            ("snapshot", " SNAP ", (140, 70, 70)),
            ("cam_switch", f" CAM: {self.camera_index} ", (60, 110, 110)),
            ("next_ep", f" EP #{self.episode_counter:02d} ", (100, 60, 140)),
            ("quit", " QUIT (Q) ", (45, 45, 160)),
        ]

        btn_count = len(btn_specs)
        margin = 8 if w >= 800 else 4
        total_margin = (btn_count + 1) * margin
        avail_w = w - total_margin
        btn_w = max(50, avail_w // btn_count)

        self.buttons = []
        cur_x = margin
        for name, label, color in btn_specs:
            rect = (cur_x, y1, cur_x + btn_w, y2)
            self.buttons.append(Button(name, label, rect, color))
            cur_x += btn_w + margin

    def _update_button_labels(self):
        """Updates button visual state depending on recording status and active camera."""
        for btn in self.buttons:
            if btn.name == "rec_stop":
                if self.is_recording:
                    btn.label = " [STOP] "
                    btn.bg_color = (30, 30, 200)  # Red
                else:
                    btn.label = " [REC] "
                    btn.bg_color = (40, 160, 40)  # Green
            elif btn.name == "pause":
                if self.is_paused:
                    btn.label = " RESUME "
                    btn.bg_color = (30, 180, 180)
                else:
                    btn.label = " PAUSE "
                    btn.bg_color = (40, 120, 180)
            elif btn.name == "cam_switch":
                btn.label = f" CAM: {self.camera_index} "
            elif btn.name == "next_ep":
                btn.label = f" EP #{self.episode_counter:02d} "

    def show_toast(self, message: str, color: Tuple[int, int, int] = (0, 220, 0), duration_sec: float = 3.5):
        """Sets a temporary on-screen notification toast."""
        self.toast_message = message
        self.toast_color = color
        self.toast_expiry = time.time() + duration_sec
        print(f"[STATUS] {message}")

    def start_recording(self):
        """Begins recording to a new timestamped file."""
        if self.is_recording:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{ts}_cam{self.camera_index}_ep{self.episode_counter:02d}.mp4"
        filepath = str(self.output_dir / filename)
        self.current_video_path = filepath

        self.async_writer = AsyncVideoWriter(
            filepath=filepath,
            width=self.actual_width,
            height=self.actual_height,
            fps=self.actual_fps if self.actual_fps > 0 else self.target_fps,
        )

        self.is_recording = True
        self.is_paused = False
        self.recording_start_time = time.time()
        self.total_paused_duration = 0.0
        self.recorded_frames = 0
        self._update_button_labels()
        self.show_toast(f"Recording started: {filename}", color=(0, 220, 0), duration_sec=2.0)

    def pause_recording(self):
        """Toggles pause/resume state."""
        if not self.is_recording:
            return
        if not self.is_paused:
            self.is_paused = True
            self.pause_start_time = time.time()
            self._update_button_labels()
            self.show_toast("Recording PAUSED", color=(0, 180, 240), duration_sec=2.0)
        else:
            self.is_paused = False
            self.total_paused_duration += (time.time() - self.pause_start_time)
            self._update_button_labels()
            self.show_toast("Recording RESUMED", color=(0, 220, 0), duration_sec=2.0)

    def stop_recording(self, auto_saved: bool = True):
        """Stops recording and finalizes video file (auto-save)."""
        if not self.is_recording:
            return

        self.is_recording = False
        self.is_paused = False
        saved_path = self.current_video_path

        if self.async_writer is not None:
            frames = self.async_writer.stop()
            self.async_writer = None
        else:
            frames = 0

        self._update_button_labels()
        action_word = "Auto-Saved" if auto_saved else "Saved"
        rel_path = os.path.relpath(saved_path) if saved_path else "unknown"
        self.show_toast(f"✓ {action_word}: {rel_path} ({frames} frames)", color=(0, 230, 80), duration_sec=4.0)

        # Automatically advance episode counter on successful stop/save
        self.episode_counter += 1
        self._update_button_labels()

    def save_explicit(self):
        """Handles explicit 'Save' button click."""
        if self.is_recording:
            self.stop_recording(auto_saved=False)
        else:
            self.show_toast("Not recording. Use [REC] or SPACE to start.", color=(0, 160, 240), duration_sec=2.5)

    def take_snapshot(self, frame: np.ndarray):
        """Saves current raw frame as high-resolution PNG."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.prefix}_{ts}_cam{self.camera_index}_ep{self.episode_counter:02d}.png"
        filepath = str(self.output_dir / filename)
        cv2.imwrite(filepath, frame)
        rel_path = os.path.relpath(filepath)
        self.show_toast(f"📸 Snapshot saved: {rel_path}", color=(220, 220, 0), duration_sec=3.0)

    def increment_episode(self):
        """Increments episode counter."""
        self.episode_counter += 1
        self._update_button_labels()
        self.show_toast(f"Episode set to #{self.episode_counter:02d}", color=(200, 120, 255), duration_sec=2.0)

    def on_mouse(self, event, x, y, flags, param):
        """Handles OpenCV mouse interaction for button clicking and hover effects."""
        self.mouse_pos = (x, y)

        # Update hover state
        for btn in self.buttons:
            btn.hovered = btn.contains(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                if btn.contains(x, y):
                    if btn.name == "rec_stop":
                        if self.is_recording:
                            self.stop_recording(auto_saved=True)
                        else:
                            self.start_recording()
                    elif btn.name == "pause":
                        self.pause_recording()
                    elif btn.name == "save":
                        self.save_explicit()
                    elif btn.name == "snapshot":
                        if hasattr(self, "_latest_raw_frame"):
                            self.take_snapshot(self._latest_raw_frame)
                    elif btn.name == "cam_switch":
                        self.cycle_camera()
                    elif btn.name == "next_ep":
                        self.increment_episode()
                    elif btn.name == "quit":
                        self._request_quit = True
                    break

    def draw_hud(self, frame: np.ndarray, current_fps: float) -> np.ndarray:
        """Overlays status header, timer, and interactive controls."""
        h, w = frame.shape[:2]
        hud_frame = frame.copy()
        font_scale = 0.44 if w >= 800 else 0.38

        # Top status bar (semi-transparent)
        top_bar_h = 38 if h >= 600 else 32
        top_overlay = hud_frame.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)
        cv2.addWeighted(top_overlay, 0.72, hud_frame, 0.28, 0, hud_frame)
        cv2.line(hud_frame, (0, top_bar_h), (w, top_bar_h), (60, 60, 60), 1)

        # Bottom control bar (semi-transparent)
        bottom_bar_h = 58 if h >= 600 else 48
        bottom_overlay = hud_frame.copy()
        cv2.rectangle(bottom_overlay, (0, h - bottom_bar_h), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(bottom_overlay, 0.75, hud_frame, 0.25, 0, hud_frame)
        cv2.line(hud_frame, (0, h - bottom_bar_h), (w, h - bottom_bar_h), (60, 60, 60), 1)

        # Draw interactive buttons
        for btn in self.buttons:
            btn.draw(hud_frame, font_scale=font_scale)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_y = int(top_bar_h * 0.68)

        # Left: Recording status / Live indicator
        if self.is_recording:
            pulse = int((time.time() * 2) % 2)
            dot_color = (0, 0, 255) if pulse else (0, 0, 160)
            cv2.circle(hud_frame, (16, top_bar_h // 2), 6, dot_color, -1)

            if self.is_paused:
                elapsed = self.pause_start_time - self.recording_start_time - self.total_paused_duration
                status_txt = f"PAUSED  {int(elapsed // 60):02d}:{elapsed % 60:04.1f} ({self.recorded_frames}f)"
                txt_color = (0, 180, 240)
            else:
                elapsed = time.time() - self.recording_start_time - self.total_paused_duration
                status_txt = f"REC  {int(elapsed // 60):02d}:{elapsed % 60:04.1f} ({self.recorded_frames}f)"
                txt_color = (0, 60, 255)

            cv2.putText(hud_frame, status_txt, (28, text_y), font, font_scale + 0.05, txt_color, 2, cv2.LINE_AA)
        else:
            cv2.circle(hud_frame, (16, top_bar_h // 2), 5, (0, 220, 0), -1)
            cv2.putText(hud_frame, "STANDBY", (28, text_y), font, font_scale, (200, 200, 200), 1, cv2.LINE_AA)

        # Center: Current run tag / episode
        center_txt = f"'{self.prefix}'  |  Ep #{self.episode_counter:02d}"
        (cw, _), _ = cv2.getTextSize(center_txt, font, font_scale, 1)
        cv2.putText(hud_frame, center_txt, ((w - cw) // 2, text_y), font, font_scale, (230, 230, 230), 1, cv2.LINE_AA)

        # Right: Camera name / index, Resolution, FPS
        cam_short = self.camera_name.split(" ")[0]
        right_txt = f"Cam {self.camera_index} ({cam_short})  |  {w}x{h}  |  {current_fps:4.1f} FPS"
        (rw, _), _ = cv2.getTextSize(right_txt, font, font_scale - 0.04, 1)
        cv2.putText(hud_frame, right_txt, (w - rw - 10, text_y), font, font_scale - 0.04, (180, 180, 180), 1, cv2.LINE_AA)

        # Active Toast notification (banner centered below header)
        now = time.time()
        if now < self.toast_expiry and self.toast_message:
            toast_w = min(w - 40, 680)
            toast_h = 32
            tx1 = (w - toast_w) // 2
            ty1 = top_bar_h + 8
            tx2 = tx1 + toast_w
            ty2 = ty1 + toast_h

            toast_overlay = hud_frame.copy()
            cv2.rectangle(toast_overlay, (tx1, ty1), (tx2, ty2), (30, 30, 30), -1)
            cv2.addWeighted(toast_overlay, 0.85, hud_frame, 0.15, 0, hud_frame)
            cv2.rectangle(hud_frame, (tx1, ty1), (tx2, ty2), self.toast_color, 1)

            (msg_w, msg_h), _ = cv2.getTextSize(self.toast_message, font, font_scale, 1)
            mx = tx1 + (toast_w - msg_w) // 2
            my = ty1 + (toast_h + msg_h) // 2
            cv2.putText(hud_frame, self.toast_message, (mx, my), font, font_scale, self.toast_color, 1, cv2.LINE_AA)

        return hud_frame

    def run(self):
        """Main camera acquisition and preview loop."""
        if self.cap is None and not self.init_camera():
            return

        self._request_quit = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        print("\n" + "=" * 64)
        print("  WIPING POLICY VIDEO RECORDER")
        print("=" * 64)
        print("  Controls:")
        print("    [SPACE] or [R] : Start / Stop recording (Auto-saves on stop)")
        print("    [TAB] or [T]   : Cycle camera index")
        print("    [P]            : Pause / Resume recording")
        print("    [S]            : Explicit Save / finalize recording")
        print("    [N]            : Next Episode (+1)")
        print("    [C]            : Take PNG snapshot")
        print("    [H]            : Toggle HUD overlay visibility")
        print("    [Q] or [ESC]   : Quit")
        print("=" * 64 + "\n")

        prev_time = time.time()
        fps_smooth = self.actual_fps if self.actual_fps > 0 else 30.0

        try:
            while not self._request_quit:
                try:
                    ret, raw_frame = self.cap.read()
                except cv2.error:
                    time.sleep(0.01)
                    continue

                if not ret or raw_frame is None:
                    time.sleep(0.01)
                    continue

                self._latest_raw_frame = raw_frame
                now = time.time()
                dt = now - prev_time
                prev_time = now
                if dt > 0:
                    current_fps = 1.0 / dt
                    fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps
                else:
                    fps_smooth = 30.0

                # Write frame to disk if recording and not paused
                if self.is_recording and not self.is_paused and self.async_writer is not None:
                    frame_to_write = raw_frame if not self.burn_hud else self.draw_hud(raw_frame, fps_smooth)
                    self.async_writer.write(frame_to_write)
                    self.recorded_frames += 1

                # Prepare display preview
                if self.show_hud:
                    display_frame = self.draw_hud(raw_frame, fps_smooth)
                else:
                    display_frame = raw_frame

                cv2.imshow(self.window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                    break
                elif key in (ord(' '), ord('r'), ord('R')):  # Space or R
                    if self.is_recording:
                        self.stop_recording(auto_saved=True)
                    else:
                        self.start_recording()
                elif key in (9, ord('t'), ord('T')):  # TAB or T: Cycle camera
                    self.cycle_camera()
                elif key in (ord('p'), ord('P')):
                    self.pause_recording()
                elif key in (ord('s'), ord('S')):
                    self.save_explicit()
                elif key in (ord('n'), ord('N')):
                    self.increment_episode()
                elif key in (ord('c'), ord('C')):
                    self.take_snapshot(raw_frame)
                elif key in (ord('h'), ord('H')):
                    self.show_hud = not self.show_hud

        finally:
            if self.is_recording:
                self.stop_recording(auto_saved=True)
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("[RECORDER] Exited cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Interactive OpenCV Video Recorder for Wiping Policy")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera index (e.g. 10, 0) or 'non-orbbec' / 'orbbec'")
    parser.add_argument("--non-orbbec", action="store_true",
                        help="Automatically select the first non-Orbbec camera device")
    parser.add_argument("--select-camera", "-s", action="store_true",
                        help="Interactive console prompt to select camera device at startup")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target recording FPS (default: 30.0)")
    parser.add_argument("--output-dir", type=str, default="recordings", help="Output directory (default: recordings)")
    parser.add_argument("--prefix", type=str, default="wiping", help="Video filename prefix (default: wiping)")
    parser.add_argument("--episode", type=int, default=1, help="Starting episode number (default: 1)")
    parser.add_argument("--burn-hud", action="store_true", help="Burn UI overlay buttons directly into saved video")
    parser.add_argument("--list-cameras", action="store_true", help="Scan and list all detected camera devices")
    args = parser.parse_args()

    # Discover working cameras
    cameras = list_available_cameras(primary_only=True)

    if args.list_cameras:
        if not cameras:
            print("No video capture devices found.")
        else:
            print(f"\nFound {len(cameras)} working camera device(s):")
            for cam in cameras:
                tag = "[Orbbec]" if cam["is_orbbec"] else "[Non-Orbbec]"
                print(f"  Camera Index {cam['index']:2d}: {cam['name']} {tag}")
                print(f"       Resolution: {cam['width']}x{cam['height']} @ {cam['fps']:.1f} FPS ({cam['backend']})")
        return

    # Determine chosen camera index
    chosen_idx = 0
    if args.select_camera:
        chosen_idx = prompt_camera_selection(cameras)
    elif args.non_orbbec or (args.camera and args.camera.lower() == "non-orbbec"):
        # Pick first non-orbbec camera
        non_orbbec = [c for c in cameras if not c["is_orbbec"]]
        if non_orbbec:
            chosen_idx = non_orbbec[0]["index"]
            print(f"[RECORDER] Selected non-Orbbec camera: Index {chosen_idx} ({non_orbbec[0]['name']})")
        else:
            print("[RECORDER] WARNING: No non-Orbbec camera found! Defaulting to first available camera.")
            chosen_idx = cameras[0]["index"] if cameras else 0
    elif args.camera is not None:
        try:
            chosen_idx = int(args.camera)
        except ValueError:
            print(f"[RECORDER] Invalid camera index '{args.camera}', defaulting to 0.")
            chosen_idx = 0
    else:
        # Default behavior: if user attached a non-Orbbec camera, default to it or 0
        non_orbbec = [c for c in cameras if not c["is_orbbec"]]
        if non_orbbec:
            chosen_idx = non_orbbec[0]["index"]
            print(f"[RECORDER] Defaulting to detected non-Orbbec camera: Index {chosen_idx} ({non_orbbec[0]['name']})")
        elif cameras:
            chosen_idx = cameras[0]["index"]
        else:
            chosen_idx = 0

    recorder = VideoRecorder(
        camera_index=chosen_idx,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=args.output_dir,
        prefix=args.prefix,
        burn_hud=args.burn_hud,
    )
    recorder.episode_counter = args.episode
    recorder.run()


if __name__ == "__main__":
    main()
