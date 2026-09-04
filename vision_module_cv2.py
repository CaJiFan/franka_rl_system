"""
vision_module_cv2.py
====================
Real-world vision front-end for the Tilted Wipe task on Franka Research 3.
Camera backend: Standard OpenCV VideoCapture (Camera Agnostic).

Replicates the Robosuite Wipe observation signals from a live RGB feed:

  Robosuite obs key        | What we compute
  -------------------------+--------------------------------------------------
  wipe_centroid            | 3D centroid of the TARGET remaining ink stroke
  wipe_radius              | Spread of the TARGET ink stroke
  proportion_wiped         | Fraction of total ink area removed since Task Start
  gripper_to_wipe_centroid | delta = centroid - eef_pos (3-vector, board frame)

Usage
-----
  python vision_module_cv2.py

Interactive controls
--------------------
  c - enter corner-calibration mode (click 4 board corners in order:
       top-left, top-right, bottom-right, bottom-left)
  t - Start Task: Registers the currently drawn strokes as 100% dirt (0% wiped).
       As the total ink area decreases, the wiped metric approaches 100%.
  s - save calibration to  calibration_cv2.npz
  l - load calibration from calibration_cv2.npz
  q - quit

Printed obs vector at 20 Hz:
  [centroid_x, centroid_y, centroid_z,  radius,  proportion_wiped,
   eef_to_centroid_x, eef_to_centroid_y, eef_to_centroid_z]

Notes
-----
- EEF position fed in via vision_state.set_eef_pos() from your ROS node.
- Segmentation: board polygon mask first -> Gaussian Blur -> Otsu on LAB-L.
- Connects to a standard UVC webcam (e.g. /dev/video0). If you have multiple
  cameras, change the `cam_index` in `run_vision_loop()`.
"""

import cv2
import numpy as np
import threading
import time
import os
import argparse

# ---------------------------------------------------------------------------
# Physical constants – adjust to your whiteboard
# ---------------------------------------------------------------------------
BOARD_WIDTH_M    = 0.60
BOARD_HEIGHT_M   = 0.45
BOARD_DIAGONAL_M = np.sqrt(BOARD_WIDTH_M**2 + BOARD_HEIGHT_M**2)

# ---------------------------------------------------------------------------
# Robot Base Calibration (Physical 3D locations of the 4 board corners)
# Measure these by jogging the robot EEF to the corners and reading (X, Y, Z)
# ---------------------------------------------------------------------------
# ROBOT_CORNERS_BASE = {
#     "TL": np.array([0.5540, -0.1587, 0.1950]), # Top-Left
#     "TR": np.array([0.5449, 0.1771, 0.1950]), # Top-Right
#     "BR": np.array([0.7530, 0.1776, 0.0388]), # Bottom-Right
#     "BL": np.array([0.7591, -0.1556, 0.0424]), # Bottom-Left
# }

# ROBOT_CORNERS_BASE (previous board position — 2026-07-31)
# ROBOT_CORNERS_BASE = {
#     "TL": np.array([0.5044, -0.1596,  0.0766]),
#     "TR": np.array([0.5345,  0.1744,  0.0701]),
#     "BR": np.array([0.7838, -0.1822, -0.0786]),
#     "BL": np.array([0.7934,  0.1682, -0.0854]),
# }

# ROBOT_CORNERS_BASE (current board position — 2026-09-01)
# Measured with 3cm printed wiping tool (tcp_offset = [0, 0, 0.030])
ROBOT_CORNERS_BASE = {
    "TL": np.array([0.4137, -0.1549,  0.1546]),  # Top-Left
    "TR": np.array([0.4162,  0.1976,  0.1517]),  # Top-Right
    "BR": np.array([0.6212,  0.1843,  0.0018]),  # Bottom-Right
    "BL": np.array([0.6185, -0.1569,  0.0051]),  # Bottom-Left
}



# Control loop target
CONTROL_HZ = 20.0

# Calibration file path (same directory as this script)
CALIB_FILE = os.path.join(os.path.dirname(__file__), "calibration_cv2.npz")

# Default segmentation params (tunable via trackbars)
DEFAULT_OTSU_OFFSET = 0    # signed offset added to Otsu threshold (-50 -> +50)
DEFAULT_MIN_AREA    = 50   # minimum contour area in pixels^2 to keep as ink
DEFAULT_MAX_AREA    = 5000 # maximum contour area (ignores huge blobs like shadows/erasers)
DEFAULT_CLOSE_RAD   = 5    # radius for morphological closing (fills holes)
TARGET_LOCK_DIST_PX = 150  # Max pixel distance to track a splitting stroke

ESC_KEY = 27


# ===========================================================================
# Board mask helpers
# ===========================================================================

def create_board_mask(corners_px, img_shape):
    if not corners_px or len(corners_px) < 4:
        return None
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts  = np.array(corners_px, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def warp_board(bgr, corners_px, out_size=400):
    if not corners_px or len(corners_px) < 4:
        return None, None
    src = np.array(corners_px, dtype=np.float32)
    dst = np.array([
        [0,        0       ],
        [out_size, 0       ],
        [out_size, out_size],
        [0,        out_size],
    ], dtype=np.float32)
    H_warp, _ = cv2.findHomography(src, dst)
    if H_warp is None:
        return None, None
    warped = cv2.warpPerspective(bgr, H_warp, (out_size, out_size))
    return warped, H_warp


# ===========================================================================
# Segmentation
# ===========================================================================

def segment_ink(bgr_img, board_mask, otsu_offset=0, min_area=50, max_area=5000, close_rad=5):
    """
    Segment black ink strokes on a white/light board.
    Returns: mask : uint8 (H,W) binary, 1 = ink, 0 = background
    """
    h, w = bgr_img.shape[:2]

    if board_mask is None:
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        L   = lab[:, :, 0]
        _, raw = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return (raw // 255).astype(np.uint8)

    # 1. Extract LAB-L channel and blur to smooth out marker texture
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L   = lab[:, :, 0].astype(np.float32)
    L   = cv2.GaussianBlur(L, (7, 7), 0)

    # 2. Adaptive Gaussian Thresholding to handle local illumination differences (shadows/glare)
    L_uint8 = np.clip(L, 0, 255).astype(np.uint8)
    
    # C is the constant subtracted from the mean. Standard value is 10.
    # The user can shift it dynamically using the trackbar (otsu_offset in [-50, 50]).
    # We map C to: 10 + otsu_offset // 2 (safe range [1, 35])
    C_val = max(1, 10 + int(otsu_offset // 2))
    
    # Block size must be odd and larger than standard stroke widths (e.g., 65 pixels)
    ink_raw = cv2.adaptiveThreshold(
        L_uint8, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        65, 
        C_val
    )

    # 3. Apply board mask (excludes arm, tool, and background outside the board)
    ink_raw = cv2.bitwise_and(ink_raw, ink_raw, mask=board_mask)

    # 4. Morphological cleanup (Close to fill holes, Open to remove specks)
    close_sz = max(1, close_rad * 2 + 1)
    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))

    ink_raw = cv2.morphologyEx(ink_raw, cv2.MORPH_CLOSE, k_close)
    ink_raw = cv2.morphologyEx(ink_raw, cv2.MORPH_OPEN,  k_open)

    # 5. Filter small and oversized blobs
    cnts, _ = cv2.findContours(ink_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros((h, w), dtype=np.uint8)
    for c in cnts:
        area = cv2.contourArea(c)
        if max(1, min_area) <= area <= max_area:
            cv2.drawContours(clean, [c], -1, 1, -1)

    return clean


# ===========================================================================
# Ink observation computation
# ===========================================================================

def compute_ink_obs(mask_01, H_board, task_start_area_px, prev_target_px=None,
                    lock_dist_px=TARGET_LOCK_DIST_PX):
    """
    Computes observation dict treating strokes as separate objects.
    Target centroid tracks the active stroke (preventing jumps when a line splits).
    Wiped metric is based on TOTAL area reduction across all objects.
    """
    cnts, _ = cv2.findContours((mask_01 * 255).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_ink_area = 0.0
    valid_contours = []

    for c in cnts:
        area = cv2.contourArea(c)
        total_ink_area += area
        if area >= 1:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_contours.append({"contour": c, "area": area, "cx": cx, "cy": cy, "M": M})

    if task_start_area_px > 0:
        proportion_wiped = float(np.clip(1.0 - total_ink_area / task_start_area_px, 0.0, 1.0))
    else:
        proportion_wiped = 0.0

    target_info = None

    if valid_contours:
        if prev_target_px is not None:
            candidates = []
            for info in valid_contours:
                dist = np.sqrt((info["cx"] - prev_target_px[0])**2 +
                               (info["cy"] - prev_target_px[1])**2)
                if dist < lock_dist_px:
                    candidates.append(info)
            if candidates:
                target_info = max(candidates, key=lambda x: x["area"])
            else:
                target_info = max(valid_contours, key=lambda x: x["area"])
        else:
            target_info = max(valid_contours, key=lambda x: x["area"])

    if target_info is None:
        return {
            "wipe_centroid":    np.zeros(3),
            "wipe_radius":      0.0,
            "proportion_wiped": proportion_wiped if task_start_area_px > 0 else 1.0,
            "centroid_px":      None,
            "ink_area_px":      0.0,
            "contours":         [info["contour"] for info in valid_contours],
            "largest_contour":  None,
        }

    M         = target_info["M"]
    cx_px     = target_info["cx"]
    cy_px     = target_info["cy"]
    mu20      = M["mu20"] / M["m00"]
    mu02      = M["mu02"] / M["m00"]
    mu11      = M["mu11"] / M["m00"]
    lam       = 0.5 * (mu20 + mu02)
    disc      = max(0.0, 0.25 * (mu20 - mu02)**2 + mu11**2)
    radius_px = np.sqrt(max(0.0, lam + np.sqrt(disc)))

    if H_board is not None:
        pt          = np.array([[[cx_px, cy_px]]], dtype=np.float32)
        pt_b        = cv2.perspectiveTransform(pt, H_board)[0, 0]
        
        # pt_b[0] is X fraction (0=Left, 1=Right)
        # pt_b[1] is Y fraction (0=Top, 1=Bottom)
        # We use Bilinear Interpolation to find the exact 3D point in the robot base frame!
        TL = ROBOT_CORNERS_BASE["TL"]
        TR = ROBOT_CORNERS_BASE["TR"]
        BR = ROBOT_CORNERS_BASE["BR"]
        BL = ROBOT_CORNERS_BASE["BL"]
        
        top_pt = TL + pt_b[0] * (TR - TL)
        bot_pt = BL + pt_b[0] * (BR - BL)
        centroid_3d = top_pt + pt_b[1] * (bot_pt - top_pt)

        H_inv   = np.linalg.inv(H_board)
        p00     = cv2.perspectiveTransform(np.array([[[0., 0.]]]), H_inv)[0, 0]
        p11     = cv2.perspectiveTransform(np.array([[[1., 1.]]]), H_inv)[0, 0]
        diag_px = max(np.linalg.norm(p11 - p00), 1.0)
        radius_m = (radius_px / diag_px) * BOARD_DIAGONAL_M
    else:
        h, w        = mask_01.shape[:2]
        # Fallback if uncalibrated (just 0s to prevent crash)
        centroid_3d = np.zeros(3)
        diag_px  = np.sqrt(h**2 + w**2)
        radius_m = (radius_px / max(diag_px, 1.0)) * BOARD_DIAGONAL_M

    wipe_radius = float(radius_m / BOARD_DIAGONAL_M)

    return {
        "wipe_centroid":    centroid_3d,
        "wipe_radius":      wipe_radius,
        "proportion_wiped": proportion_wiped,
        "centroid_px":      (cx_px, cy_px),
        "ink_area_px":      total_ink_area,
        "contours":         [info["contour"] for info in valid_contours],
        "largest_contour":  target_info["contour"],
    }


def build_obs_vector(ink_obs, eef_pos=None):
    centroid = ink_obs["wipe_centroid"]
    g2c      = centroid - np.asarray(eef_pos) if eef_pos is not None else np.zeros(3)
    return np.concatenate([
        centroid,
        [ink_obs["wipe_radius"]],
        [ink_obs["proportion_wiped"]],
        g2c,
    ]).astype(np.float32)


# ===========================================================================
# Calibration
# ===========================================================================

class BoardCalibrator:
    def __init__(self):
        self.corners_px = []
        self.H          = None
        self.active     = False
        # Reference to MarkerPlacer so we can forward non-calibration clicks
        self.marker_placer = None

    def start_calibration(self):
        self.corners_px = []
        self.active     = True
        print("\n[CAL] Click the 4 board corners in order:")
        print("       1=TL  2=TR  3=BR  4=BL")

    def on_mouse(self, event, x, y, flags, param):
        if self.active:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.corners_px.append((x, y))
                if len(self.corners_px) == 4:
                    self._compute_homography()
                    self.active = False
        elif self.marker_placer is not None:
            # Forward clicks to marker placer when not calibrating
            self.marker_placer.on_mouse(event, x, y, flags, param)

    def _compute_homography(self):
        src       = np.array(self.corners_px, dtype=np.float32)
        dst       = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)
        print("[CAL] Homography computed v")

    def save(self, path=CALIB_FILE):
        if self.H is not None:
            np.savez(path, H=self.H, corners_px=np.array(self.corners_px))
            print(f"[CAL] Saved -> {path}")

    def load(self, path=CALIB_FILE):
        if os.path.isfile(path):
            d               = np.load(path)
            self.H          = d["H"]
            self.corners_px = d["corners_px"].tolist()
            print(f"[CAL] Loaded -- corners: {self.corners_px}")


# ===========================================================================
# Manual Marker Placer
# ===========================================================================

class MarkerPlacer:
    """
    Lets the user click up to MAX_MARKERS positions on the camera image.
    Positions are stored as pixel coordinates and later converted to 3-D
    robot-frame points by MarkerTracker.initialize().

    Controls:
      M          – toggle placement mode on/off
      Left-click – place next marker (only when mode is ON)
      R          – clear all placed markers
    """
    MAX_MARKERS = 5

    def __init__(self):
        self.active: bool       = False   # placement mode toggle
        self.points_px: list    = []      # list of (x, y) pixel tuples

    def toggle(self):
        self.active = not self.active
        state = "ON" if self.active else "OFF"
        print(f"[PLACER] Marker placement mode {state}. "
              f"{'Click on the board to place waypoints.' if self.active else ''}")

    def reset(self):
        self.points_px = []
        self.active    = False
        print("[PLACER] All manual markers cleared.")

    def on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points_px) >= self.MAX_MARKERS:
                print(f"[PLACER] Max {self.MAX_MARKERS} markers already placed. "
                      "Press R to clear.")
                return
            self.points_px.append((x, y))
            idx = len(self.points_px) - 1
            print(f"[PLACER] Placed M{idx} at pixel ({x}, {y})  "
                  f"({len(self.points_px)}/{self.MAX_MARKERS})")
            if len(self.points_px) == self.MAX_MARKERS:
                self.active = False
                print("[PLACER] 5 markers placed. Press T to start task.")


# ===========================================================================

class VisionState:
    def __init__(self):
        self._lock         = threading.Lock()
        self._obs_vec      = np.zeros(8, dtype=np.float32)
        self._eef_pos      = None
        self._eef_quat     = None
        # Per-marker data: list of 5 dicts {centroid_3d: np[3], wiped: float}
        # Zero-padded to always have exactly 5 entries.
        self._marker_states = [
            {"centroid_3d": np.zeros(3, dtype=np.float32), "wiped": 1.0}
            for _ in range(5)
        ]

    def set_eef_pose(self, pos, quat):
        with self._lock:
            self._eef_pos  = np.asarray(pos, dtype=np.float32).copy()
            self._eef_quat = np.asarray(quat, dtype=np.float32).copy()

    def set_marker_states(self, states):
        """states: list of up to 5 dicts with 'centroid_3d' (np[3]) and 'wiped' (float).
        Always padded to exactly 5 entries (wiped=1.0 for undetected slots)."""
        with self._lock:
            padded = []
            for i in range(5):
                if i < len(states):
                    padded.append({
                        "centroid_3d": np.asarray(states[i]["centroid_3d"], dtype=np.float32).copy(),
                        "wiped": float(states[i]["wiped"]),
                    })
                else:
                    padded.append({"centroid_3d": np.zeros(3, dtype=np.float32), "wiped": 1.0})
            self._marker_states = padded

    def get_marker_states(self):
        """Returns a list of 5 dicts with 'centroid_3d' and 'wiped'."""
        with self._lock:
            return [
                {"centroid_3d": m["centroid_3d"].copy(), "wiped": m["wiped"]}
                for m in self._marker_states
            ]

    def update(self, obs_vec):
        with self._lock:
            self._obs_vec = obs_vec.copy()

    def get_obs(self):
        with self._lock:
            return self._obs_vec.copy()

    def get_eef_pose(self):
        with self._lock:
            if self._eef_pos is None or self._eef_quat is None:
                return None, None
            return self._eef_pos.copy(), self._eef_quat.copy()


vision_state = VisionState()


# ===========================================================================
# Multi-Marker Tracker
# ===========================================================================

class MarkerTracker:
    """
    Tracks up to 5 individual marker blobs on the whiteboard.

    Lifecycle:
      initialize(mask_01, H_board) — called on T press: detects blobs,
          records each marker's centroid + area, sorts by robot Y (ascending).
      update(mask_01) — called each frame: re-measures each marker's remaining
          area inside a circular search window; marks wiped when < WIPE_RATIO.
      get_marker_states() — returns list of 5 zero-padded dicts.
    """
    MAX_MARKERS  = 5
    WIPE_RATIO   = 0.35   # marker counted wiped when area < 35 % of start
    SEARCH_R_PX  = 25     # pixel radius (2.5 cm) to re-detect each marker per frame

    def __init__(self):
        self.markers: list[dict] = []  # {cx_px, cy_px, centroid_3d, start_area, current_area, wiped}
        self.total_start_area: float = 0.0
        self.initialized: bool = False

    # ------------------------------------------------------------------
    @property
    def active_idx(self) -> int:
        for i, m in enumerate(self.markers):
            if not m["wiped"]:
                return i
        return len(self.markers)

    @property
    def active_marker(self):
        idx = self.active_idx
        return self.markers[idx] if idx < len(self.markers) else None

    @property
    def proportion_wiped(self) -> float:
        if self.total_start_area <= 0:
            return 0.0
        remaining = sum(m["current_area"] for m in self.markers)
        return float(np.clip(1.0 - remaining / self.total_start_area, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _px_to_3d(self, cx_px, cy_px, H_board):
        """Map image pixel to robot-frame 3D via homography + bilinear interp."""
        pt    = np.array([[[cx_px, cy_px]]], dtype=np.float32)
        pt_b  = cv2.perspectiveTransform(pt, H_board)[0, 0]
        TL = ROBOT_CORNERS_BASE["TL"]
        TR = ROBOT_CORNERS_BASE["TR"]
        BR = ROBOT_CORNERS_BASE["BR"]
        BL = ROBOT_CORNERS_BASE["BL"]
        top_pt = TL + pt_b[0] * (TR - TL)
        bot_pt = BL + pt_b[0] * (BR - BL)
        return (top_pt + pt_b[1] * (bot_pt - top_pt)).astype(np.float32)

    def _area_in_neighborhood(self, mask_01, cx_px, cy_px):
        """Count ink pixels within SEARCH_R_PX of (cx_px, cy_px)."""
        h, w = mask_01.shape[:2]
        # Build circular neighbourhood mask
        ys, xs = np.ogrid[:h, :w]
        circle = (xs - cx_px) ** 2 + (ys - cy_px) ** 2 <= self.SEARCH_R_PX ** 2
        return float(np.count_nonzero(mask_01 & circle))

    # ------------------------------------------------------------------
    def initialize(self, mask_01, H_board, manual_px=None):
        """
        Initialize marker tracking.

        If manual_px is provided (list of (x, y) pixel tuples from MarkerPlacer),
        those positions are used directly as marker centroids — no blob detection.
        Otherwise falls back to automatic blob detection via findContours.

        In both cases markers are sorted by robot Y (ascending).
        """
        self.markers = []
        self.total_start_area = 0.0
        self.initialized = False

        if H_board is None:
            print("[MARKER] Cannot initialise: no homography calibration found.")
            return

        blobs = []

        if manual_px:
            # ── Manual mode: use clicked pixel positions ──────────────────────
            print(f"[MARKER] Using {len(manual_px)} manually-placed marker(s).")
            for (px, py) in manual_px[:self.MAX_MARKERS]:
                # Measure ink area around this click point for wipe detection
                area = self._area_in_neighborhood(mask_01, px, py)
                blobs.append({"cx_px": px, "cy_px": py, "area": max(area, 1.0)})
        else:
            # ── Auto mode: detect blobs from ink mask ─────────────────────────
            print("[MARKER] Auto-detecting blobs from ink mask...")
            cnts, _ = cv2.findContours(
                (mask_01 * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 20:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                blobs.append({"cx_px": cx, "cy_px": cy, "area": area})

            if not blobs:
                print("[MARKER] No blobs detected — draw markers or use M+click to place manually!")
                return

        # Compute 3D centroid for each blob/click
        for b in blobs:
            b["centroid_3d"] = self._px_to_3d(b["cx_px"], b["cy_px"], H_board)

        # Sort by robot Y ascending (sim convention: lowest Y first)
        blobs.sort(key=lambda b: b["centroid_3d"][1])
        blobs = blobs[:self.MAX_MARKERS]

        for b in blobs:
            self.markers.append({
                "cx_px":       b["cx_px"],
                "cy_px":       b["cy_px"],
                "centroid_3d": b["centroid_3d"],
                "start_area":  b["area"],
                "current_area": b["area"],
                "wiped":       False,
            })
            self.total_start_area += b["area"]

        self.initialized = True
        positions = [(m["centroid_3d"][0], m["centroid_3d"][1], m["centroid_3d"][2])
                     for m in self.markers]
        print(f"[MARKER] Initialized {len(self.markers)} marker(s), sorted by Y:")
        for i, p in enumerate(positions):
            print(f"  M{i}: X={p[0]:+.3f}  Y={p[1]:+.3f}  Z={p[2]:+.3f}")

    def update(self, mask_01, raw_mask_01=None, eef_pos_eraser=None, contact_active=False):
        """Re-measure active marker's area; advance active index when wiped."""
        if not self.initialized:
            return

        # Contact & Proximity Validation Guard:
        # A marker can ONLY transition to WIPED if the robot is physically near the surface
        # (within 4.5 cm of local tilted board plane) or in active contact (|Fz| > 2N).
        # This prevents camera line-of-sight occlusion by the Franka arm body from falsely triggering wipes in mid-air.
        z_surface_local = 0.1546 - 0.727 * (eef_pos_eraser[0] - 0.4137) if eef_pos_eraser is not None else 0.05
        near_surface = (
            contact_active or 
            (eef_pos_eraser is not None and abs(eef_pos_eraser[2] - z_surface_local) < 0.045)
        )

        # When near_surface is True (robot is physically touching/wiping board), use mask_01
        # which incorporates the digital eraser mask to instantly validate the wipe under the felt pad.
        # When near_surface is False (robot hovering in mid-air), use raw_mask_01 to prevent mid-air occlusion false-wipes.
        eval_mask = mask_01 if near_surface else (raw_mask_01 if raw_mask_01 is not None else mask_01)

        for i, m in enumerate(self.markers):
            if m["wiped"]:
                m["current_area"] = 0.0
                continue

            # Sequential Lock: ONLY evaluate wiping for the CURRENT active marker (or earlier).
            # Prevents adjacent future markers (M3, M4) from being prematurely marked as wiped
            # when the 90px eraser mask overlaps them while wiping M2!
            if i > self.active_idx:
                continue

            area = self._area_in_neighborhood(eval_mask, m["cx_px"], m["cy_px"])
            m["current_area"] = area

            # Wiping Trigger:
            # 1. Normal ink wiping: area < start_area * 35%
            # 2. Blank spot fallback: if a marker was placed on a blank/low-ink area (start_area < 30 px),
            #    immediately mark it as wiped upon robot arrival (near_surface == True).
            is_wiped = (
                (m["start_area"] >= 30.0 and area < m["start_area"] * self.WIPE_RATIO) or
                (m["start_area"] < 30.0)
            )
            if near_surface and is_wiped:
                m["wiped"] = True
                print(f"[MARKER] M{i} wiped! (area={area:.0f} px, start_area={m['start_area']:.0f} px)")

    def force_wipe_active(self):
        """Manually mark the current active marker as wiped (triggered by 'N' key press)."""
        idx = self.active_idx
        if idx < len(self.markers):
            self.markers[idx]["wiped"] = True
            nxt = f"M{idx+1}" if idx + 1 < len(self.markers) else "ALL DONE"
            print(f"\n[MANUAL OVERRIDE] Manually marked M{idx} as WIPED! Advancing to {nxt}...")
            return True
        print(f"\n[MANUAL OVERRIDE] All markers already wiped!")
        return False

    def get_marker_states(self) -> list:
        """Returns exactly 5 dicts (zero-padded). Used by VisionState.set_marker_states."""
        states = []
        for i in range(self.MAX_MARKERS):
            if i < len(self.markers):
                m = self.markers[i]
                states.append({
                    "centroid_3d": m["centroid_3d"],
                    "wiped": 1.0 if m["wiped"] else 0.0,
                })
            else:
                # Undetected slot: zero position, treated as already wiped
                states.append({"centroid_3d": np.zeros(3, dtype=np.float32), "wiped": 1.0})
        return states


# ===========================================================================
# Main live-feed loop
# ===========================================================================

def nothing(x):
    pass


def run_vision_loop(half_frame=False, cam_index=0):
    # ── Camera init ──────────────────────────────────────────────────────────
    # Force V4L2 backend for Linux stability
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"[VIS] Failed to open camera at index {cam_index}.")
        print("      Try changing `cam_index` in the script if you have multiple cameras.")
        return

    # Force YUYV format (uncompressed) at a lower 15 FPS to prevent USB bandwidth crashes
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
    print(f"[CAM] Camera opened successfully via OpenCV: {actual_w}x{actual_h} @ {actual_fps}fps ({codec_str})")

    calibrator = BoardCalibrator()
    calibrator.load()

    marker_placer = MarkerPlacer()
    calibrator.marker_placer = marker_placer   # forward non-calib clicks

    win = "Wipe Vision Module - OpenCV Generic"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, calibrator.on_mouse)

    cv2.createTrackbar("Otsu Offset  (-50->+50)", win, 50 + DEFAULT_OTSU_OFFSET, 100, nothing)
    cv2.createTrackbar("Min Area (px2)",           win, DEFAULT_MIN_AREA,         300, nothing)
    cv2.createTrackbar("Max Area (px2)",           win, DEFAULT_MAX_AREA,         30000, nothing)
    cv2.createTrackbar("Close Radius (holes)",     win, DEFAULT_CLOSE_RAD,        20,  nothing)

    marker_tracker   = MarkerTracker()
    task_started     = False
    locked_target_px = None

    print("\n[VIS] Controls:")
    print("   c - calibrate corners  |  t - Start Task")
    print("   m - toggle marker placement mode (then click board to place waypoints)")
    print("   r - reset / clear placed markers")
    print("   s - save calib         |  l - load calib")
    print("   q / ESC - quit")
    print("   TIP: Place markers manually with M+click, then press T to start.\n")

    period   = 1.0 / CONTROL_HZ
    t_period = time.time()
    
    try:
        while True:
            # ── Frame acquisition ─────────────────────────────────────────────
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    cv2.waitKey(1)
                    continue
            except cv2.error:
                # Catch corrupted frames due to USB drops
                cv2.waitKey(1)
                continue

            # Flip the frame 180 degrees because the camera is mounted upside down
            frame = cv2.flip(frame, -1)

            # The ZED (and some other stereo webcams) outputs a side-by-side frame (Left + Right).
            # We crop the frame in half to just use the left camera feed if the --half flag is provided:
            if half_frame:
                h, w = frame.shape[:2]
                bgr = frame[:, :w//2]
            else:
                bgr = frame

            # ── Trackbar params ───────────────────────────────────────────────
            otsu_offset = cv2.getTrackbarPos("Otsu Offset  (-50->+50)", win) - 50
            min_area    = max(1, cv2.getTrackbarPos("Min Area (px2)", win))
            max_area    = max(min_area + 1, cv2.getTrackbarPos("Max Area (px2)", win))
            close_rad   = cv2.getTrackbarPos("Close Radius (holes)", win)

            # ── Processing ───────────────────────────────────────────────────
            board_mask = create_board_mask(calibrator.corners_px, bgr.shape)

            # Mask out the end-effector area and the eraser tool to avoid self-segmentation
            eef_pos, eef_quat = vision_state.get_eef_pose()
            if eef_pos is not None and eef_quat is not None and calibrator.H is not None and board_mask is not None:
                try:
                    from scipy.spatial.transform import Rotation as R
                    TL = ROBOT_CORNERS_BASE["TL"]
                    TR = ROBOT_CORNERS_BASE["TR"]
                    BL = ROBOT_CORNERS_BASE["BL"]
                    
                    vx = TR - TL
                    vy = BL - TL
                    
                    # Compute eraser position
                    r_curr = R.from_quat(eef_quat)
                    tcp_offset = np.array([0.0, 0.0, 0.185])
                    eef_pos_eraser = eef_pos + r_curr.apply(tcp_offset)
                    
                    # 1. Project Flange center
                    v = eef_pos - TL
                    x_frac = np.dot(v, vx) / np.dot(vx, vx)
                    y_frac = np.dot(v, vy) / np.dot(vy, vy)
                    
                    H_inv = np.linalg.inv(calibrator.H)
                    pt = np.array([[[x_frac, y_frac]]], dtype=np.float32)
                    pt_camera = cv2.perspectiveTransform(pt, H_inv)[0, 0]
                    eef_cx, eef_cy = int(pt_camera[0]), int(pt_camera[1])
                    
                    # 2. Project Eraser center
                    v_eraser = eef_pos_eraser - TL
                    x_frac_eraser = np.dot(v_eraser, vx) / np.dot(vx, vx)
                    y_frac_eraser = np.dot(v_eraser, vy) / np.dot(vy, vy)
                    pt_eraser = np.array([[[x_frac_eraser, y_frac_eraser]]], dtype=np.float32)
                    pt_camera_eraser = cv2.perspectiveTransform(pt_eraser, H_inv)[0, 0]
                    eraser_cx, eraser_cy = int(pt_camera_eraser[0]), int(pt_camera_eraser[1])
                    
                    # Ensure coordinates are within image bounds
                    h, w = board_mask.shape[:2]
                    
                    # Mask flange (45px radius)
                    if 0 <= eef_cx < w and 0 <= eef_cy < h:
                        cv2.circle(board_mask, (eef_cx, eef_cy), 45, 0, -1)
                        
                        # (Cable mask line disabled to prevent digitally erasing ink on the board)
                        # if len(calibrator.corners_px) > 0:
                        #     TL_px = calibrator.corners_px[0]
                        #     cv2.line(board_mask, (eef_cx, eef_cy), (int(TL_px[0]), int(TL_px[1])), 0, 80)
                            
                    # Mask physical eraser tool assembly (90px radius around the eraser tip)
                    # ONLY apply this mask when the eraser is close to the board (Z < 0.22) to avoid parallax errors when high up!
                    if eef_pos_eraser[2] < 0.22:
                        if 0 <= eraser_cx < w and 0 <= eraser_cy < h:
                            cv2.circle(board_mask, (eraser_cx, eraser_cy), 90, 0, -1)
                except Exception as e:
                    pass

            mask_01 = segment_ink(
                bgr,
                board_mask=board_mask,
                otsu_offset=otsu_offset,
                min_area=min_area,
                max_area=max_area,
                close_rad=close_rad,
            )

            ink_obs = compute_ink_obs(
                mask_01,
                calibrator.H,
                marker_tracker.total_start_area,  # was task_start_area_px
                prev_target_px=locked_target_px,
                lock_dist_px=TARGET_LOCK_DIST_PX,
            )
            locked_target_px = ink_obs["centroid_px"]

            raw_mask_01 = segment_ink(
                bgr,
                board_mask=create_board_mask(calibrator.corners_px, bgr.shape),
                otsu_offset=otsu_offset,
                min_area=min_area,
                max_area=max_area,
                close_rad=close_rad,
            )

            # ── Update marker tracker each frame ─────────────────────────────
            if task_started and marker_tracker.initialized:
                marker_tracker.update(
                    mask_01, 
                    raw_mask_01=raw_mask_01,
                    eef_pos_eraser=eef_pos_eraser if 'eef_pos_eraser' in locals() else None
                )

            # Push per-marker states to shared VisionState for ROS publishing
            vision_state.set_marker_states(marker_tracker.get_marker_states())

            # Console log @ CONTROL_HZ
            eef_pos, _ = vision_state.get_eef_pose()
            obs_vec = build_obs_vector(ink_obs, eef_pos)
            vision_state.update(obs_vec)

            now = time.time()
            if now - t_period >= period:
                t_period = now
                am = marker_tracker.active_marker
                active_str = f"M{marker_tracker.active_idx}" if am else "ALL DONE"
                print(
                    f"\r[MARKER] Active={active_str}  "
                    f"wiped={marker_tracker.proportion_wiped*100:5.1f}%  "
                    f"strokes={len(ink_obs.get('contours',[]))}  "
                    f"area={ink_obs['ink_area_px']:.0f}px",
                    end="", flush=True,
                )

            # ── Visualisation ─────────────────────────────────────────────────
            vis         = bgr.copy()
            ink_display = np.zeros_like(vis)
            ink_display[:, :, 2] = (mask_01 * 255).astype(np.uint8)
            vis = cv2.addWeighted(vis, 0.65, ink_display, 0.6, 0)

            if board_mask is not None and len(calibrator.corners_px) == 4:
                pts = np.array(calibrator.corners_px, dtype=np.int32)
                cv2.polylines(vis, [pts], isClosed=True, color=(0, 200, 255), thickness=2)

            if ink_obs.get("contours"):
                cv2.drawContours(vis, ink_obs["contours"], -1, (255, 200, 0), 1)

            # ── Per-marker visualization ──────────────────────────────────────
            MARKER_COLORS = {
                "active":  (0,   255, 0),    # green
                "pending": (0,   165, 255),  # orange
                "wiped":   (120, 120, 120),  # grey
            }
            if marker_tracker.initialized:
                for mi, m in enumerate(marker_tracker.markers):
                    # Use the stored pixel position directly — no back-projection needed
                    px, py = int(m["cx_px"]), int(m["cy_px"])
                    if mi == marker_tracker.active_idx:
                        color = MARKER_COLORS["active"]
                        cv2.circle(vis, (px, py), 18, color, 3)
                        cv2.drawMarker(vis, (px, py), color, cv2.MARKER_CROSS, 36, 2)
                    elif m["wiped"]:
                        color = MARKER_COLORS["wiped"]
                        cv2.circle(vis, (px, py), 16, color, 2)
                        cv2.putText(vis, "v", (px - 5, py + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        color = MARKER_COLORS["pending"]
                        cv2.circle(vis, (px, py), 16, color, 2)
                    cv2.putText(vis, f"M{mi}", (px - 8, py - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


            if eef_pos is not None and 'eef_cx' in locals() and 'eef_cy' in locals():
                # Draw the EEF circular mask
                cv2.circle(vis, (eef_cx, eef_cy), 45, (255, 120, 0), 2)
                cv2.putText(vis, "EEF MASK", (eef_cx - 20, eef_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1)
                
                # Draw the physical eraser mask if active (Z < 0.22)
                if 'eraser_cx' in locals() and 'eraser_cy' in locals() and 'eef_pos_eraser' in locals():
                    if eef_pos_eraser[2] < 0.22:
                        cv2.circle(vis, (eraser_cx, eraser_cy), 90, (255, 120, 0), 2)
                        cv2.putText(vis, "ERASER MASK", (eraser_cx - 35, eraser_cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 120, 0), 1)
                
                # Draw the cable line mask visual (disabled)
                # if len(calibrator.corners_px) > 0:
                #     TL_px = calibrator.corners_px[0]
                #     overlay = vis.copy()
                #     cv2.line(overlay, (eef_cx, eef_cy), (int(TL_px[0]), int(TL_px[1])), (255, 120, 0), 80)
                #     cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

            if calibrator.active:
                for (px, py) in calibrator.corners_px:
                    cv2.circle(vis, (px, py), 6, (0, 140, 255), -1)

            # ── Pending manual marker positions (before T) ────────────────────
            if not task_started and marker_placer.points_px:
                for mi, (px, py) in enumerate(marker_placer.points_px):
                    col = (255, 255, 0)  # cyan
                    cv2.circle(vis, (px, py), 16, col, 2)
                    cv2.drawMarker(vis, (px, py), col, cv2.MARKER_CROSS, 28, 2)
                    cv2.putText(vis, f"M{mi}", (px - 8, py - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            if marker_placer.active:
                cv2.putText(vis, f"[PLACE MODE] Click board to place M{len(marker_placer.points_px)}",
                            (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            # Per-marker status string: M0✓ M1● M2○ ...
            if marker_tracker.initialized:
                mstatus = []
                for mi, m in enumerate(marker_tracker.markers):
                    sym = "✓" if m["wiped"] else ("●" if mi == marker_tracker.active_idx else "○")
                    mstatus.append(f"M{mi}{sym}")
                # Pad undetected slots
                for mi in range(len(marker_tracker.markers), 5):
                    mstatus.append(f"M{mi}-")
                marker_hud = "  ".join(mstatus)
                am = marker_tracker.active_marker
                act_pos = f"({am['centroid_3d'][0]:+.3f},{am['centroid_3d'][1]:+.3f})" if am else "DONE"
            else:
                marker_hud = "Press T to start"
                act_pos = "---"

            hud = [
                f"Markers: {marker_hud}",
                f"Active:  {act_pos} m  |  Wiped: {marker_tracker.proportion_wiped*100:5.1f}%",
                f"Total Ink Area: {ink_obs['ink_area_px']:.0f} px  ({'TASK RUNNING' if task_started else 'press T to start'})",
                f"Camera: OpenCV Generic  [Controls: T=Start | N=Skip Marker | R=Reset]",
                f"--- Trackbars ---",
                f"Otsu Offset: {otsu_offset} | Min Area: {min_area} px2 | Max Area: {max_area} px2 | Close Rad: {close_rad}",
            ]
            for i, line in enumerate(hud):
                cv2.putText(vis, line, (10, 26 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 1, cv2.LINE_AA)

            warped_bgr, H_warp = warp_board(bgr, calibrator.corners_px, out_size=300)
            if warped_bgr is not None and H_warp is not None:
                mask_vis  = (mask_01 * 255).astype(np.uint8)
                warp_mask = cv2.warpPerspective(mask_vis, H_warp, (300, 300))
                warp_vis  = warped_bgr.copy()
                warp_ink  = np.zeros_like(warp_vis)
                warp_ink[:, :, 2] = warp_mask
                warp_vis = cv2.addWeighted(warp_vis, 0.6, warp_ink, 0.7, 0)
                cv2.putText(warp_vis, "Board (warped)", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                h_vis = vis.shape[0]
                pad   = h_vis - 300
                if pad > 0:
                    warp_vis = cv2.copyMakeBorder(warp_vis, 0, pad, 0, 0,
                                                  cv2.BORDER_CONSTANT, value=(30, 30, 30))
                elif pad < 0:
                    warp_vis = warp_vis[:h_vis]
                combined = np.hstack([vis, warp_vis])
            else:
                combined = vis

            max_w  = 1600
            cw     = combined.shape[1]
            scale  = min(1.0, max_w / cw)
            disp   = cv2.resize(combined, (int(cw * scale), int(combined.shape[0] * scale)))
            cv2.imshow(win, disp)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ESC_KEY):
                break
            elif key == ord('c'):
                calibrator.start_calibration()
            elif key == ord('m'):
                marker_placer.toggle()
            elif key == ord('r'):
                marker_placer.reset()
                marker_tracker.initialized = False  # also reset live tracker
                marker_tracker.markers = []
                task_started = False
                print("[VIS] Task reset. Place new markers with M+click, then press T.")
            elif key == ord('t'):
                # Use manually-placed markers if available, else auto-detect
                manual = marker_placer.points_px if marker_placer.points_px else None
                marker_tracker.initialize(mask_01, calibrator.H, manual_px=manual)
                task_started = True
                mode = "manual" if manual else "auto"
                print(f"\n[VIS] TASK STARTED ({mode}). {len(marker_tracker.markers)} marker(s).")
            elif key in (ord('n'), ord('N')):
                marker_tracker.force_wipe_active()
            elif key == ord('s'):
                calibrator.save()
            elif key == ord('l'):
                calibrator.load()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[VIS] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wipe Vision Module - OpenCV Generic")
    parser.add_argument("--half", action="store_true", help="Crop the camera frame in half (useful for stereo side-by-side feeds).")
    parser.add_argument("--cam_index", type=int, default=2, help="OpenCV Camera Index. Default is 2.")
    args = parser.parse_args()

    run_vision_loop(half_frame=args.half, cam_index=args.cam_index)
