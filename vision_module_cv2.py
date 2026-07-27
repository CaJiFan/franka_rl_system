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

    # 2. Compute Otsu threshold using ONLY board pixels
    board_pixels = L[board_mask > 0]
    if len(board_pixels) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    board_uint8 = board_pixels.astype(np.uint8).reshape(-1, 1)
    otsu_val, _ = cv2.threshold(board_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply offset
    thresh = float(otsu_val) - float(otsu_offset)
    thresh = np.clip(thresh, 1, 254)

    # 3. Threshold: pixels DARKER than thresh -> ink
    ink_raw = (L < thresh).astype(np.uint8) * 255

    # 4. Apply board mask
    ink_raw = cv2.bitwise_and(ink_raw, ink_raw, mask=board_mask)

    # 5. Morphological cleanup (Close to fill holes, Open to remove specks)
    close_sz = max(1, close_rad * 2 + 1)
    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_sz, close_sz))

    ink_raw = cv2.morphologyEx(ink_raw, cv2.MORPH_CLOSE, k_close)
    ink_raw = cv2.morphologyEx(ink_raw, cv2.MORPH_OPEN,  k_open)

    # 6. Filter small and oversized blobs
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
        centroid_3d = np.array([
            pt_b[0] * BOARD_WIDTH_M  - BOARD_WIDTH_M  / 2.0,
            pt_b[1] * BOARD_HEIGHT_M - BOARD_HEIGHT_M / 2.0,
            0.0,
        ])
        H_inv   = np.linalg.inv(H_board)
        p00     = cv2.perspectiveTransform(np.array([[[0., 0.]]]), H_inv)[0, 0]
        p11     = cv2.perspectiveTransform(np.array([[[1., 1.]]]), H_inv)[0, 0]
        diag_px = max(np.linalg.norm(p11 - p00), 1.0)
        radius_m = (radius_px / diag_px) * BOARD_DIAGONAL_M
    else:
        h, w        = mask_01.shape[:2]
        centroid_3d = np.array([
            (cx_px / max(w, 1) - 0.5) * BOARD_WIDTH_M,
            (cy_px / max(h, 1) - 0.5) * BOARD_HEIGHT_M,
            0.0,
        ])
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

    def start_calibration(self):
        self.corners_px = []
        self.active     = True
        print("\n[CAL] Click the 4 board corners in order:")
        print("       1=TL  2=TR  3=BR  4=BL")

    def on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.corners_px.append((x, y))
            if len(self.corners_px) == 4:
                self._compute_homography()
                self.active = False

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
# Thread-safe shared state
# ===========================================================================

class VisionState:
    def __init__(self):
        self._lock    = threading.Lock()
        self._obs_vec = np.zeros(8, dtype=np.float32)
        self._eef_pos = None

    def set_eef_pos(self, pos):
        with self._lock:
            self._eef_pos = np.asarray(pos, dtype=np.float32).copy()

    def update(self, obs_vec):
        with self._lock:
            self._obs_vec = obs_vec.copy()

    def get_obs(self):
        with self._lock:
            return self._obs_vec.copy()

    def get_eef_pos(self):
        with self._lock:
            return self._eef_pos.copy() if self._eef_pos is not None else None


vision_state = VisionState()


# ===========================================================================
# Main live-feed loop
# ===========================================================================

def nothing(x):
    pass


def run_vision_loop(half_frame=False):
    # ── Camera init ──────────────────────────────────────────────────────────
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"[VIS] Failed to open camera at index {cam_index}.")
        print("      Try changing `cam_index` in the script if you have multiple cameras.")
        return

    # Typical webcam defaults
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[CAM] Camera opened successfully via OpenCV: {actual_w}x{actual_h}")

    calibrator = BoardCalibrator()
    calibrator.load()

    win = "Wipe Vision Module - OpenCV Generic"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, calibrator.on_mouse)

    cv2.createTrackbar("Otsu Offset  (-50->+50)", win, 50 + DEFAULT_OTSU_OFFSET, 100, nothing)
    cv2.createTrackbar("Min Area (px2)",           win, DEFAULT_MIN_AREA,         300, nothing)
    cv2.createTrackbar("Max Area (px2)",           win, DEFAULT_MAX_AREA,         30000, nothing)
    cv2.createTrackbar("Close Radius (holes)",     win, DEFAULT_CLOSE_RAD,        20,  nothing)

    task_start_area_px = 0.0
    task_started       = False
    locked_target_px   = None

    print("\n[VIS] Controls:")
    print("   c - calibrate corners  |  t - Start Task (registers ink as 0% wiped)")
    print("   s - save calib         |  l - load calib")
    print("   q / ESC - quit\n")

    period   = 1.0 / CONTROL_HZ
    t_period = time.time()
    
    try:
        while True:
            # ── Frame acquisition ─────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret or frame is None:
                cv2.waitKey(1)
                continue

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
                task_start_area_px,
                prev_target_px=locked_target_px,
                lock_dist_px=TARGET_LOCK_DIST_PX,
            )
            locked_target_px = ink_obs["centroid_px"]

            eef_pos = vision_state.get_eef_pos()
            obs_vec = build_obs_vector(ink_obs, eef_pos)
            vision_state.update(obs_vec)

            # ── Console log @ CONTROL_HZ ──────────────────────────────────────
            now = time.time()
            if now - t_period >= period:
                t_period = now
                c   = ink_obs["wipe_centroid"]
                g2c = obs_vec[5:8]
                print(
                    f"\r[OBS] "
                    f"cent=({c[0]:+.3f},{c[1]:+.3f})m  "
                    f"wiped={ink_obs['proportion_wiped']*100:5.1f}%  "
                    f"eef->c=({g2c[0]:+.3f},{g2c[1]:+.3f},{g2c[2]:+.3f})  "
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

            if ink_obs["centroid_px"] is not None:
                cx, cy = ink_obs["centroid_px"]
                if ink_obs.get("largest_contour") is not None:
                    cv2.drawContours(vis, [ink_obs["largest_contour"]], -1, (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 12, (0, 255, 0), 2)
                cv2.drawMarker(vis, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)

            if calibrator.active:
                for (px, py) in calibrator.corners_px:
                    cv2.circle(vis, (px, py), 6, (0, 140, 255), -1)

            hud = [
                f"Wiped: {ink_obs['proportion_wiped']*100:5.1f}%  |  Strokes: {len(ink_obs.get('contours',[]))}",
                f"Target Centroid: ({ink_obs['wipe_centroid'][0]:+.3f}, {ink_obs['wipe_centroid'][1]:+.3f}) m",
                f"Total Ink Area: {ink_obs['ink_area_px']:.0f} px",
                f"Task Start Area: {task_start_area_px:.0f} px  ({'SET v' if task_started else 'NOT SET - press T'})",
                f"EEF->Cent: ({obs_vec[5]:+.3f}, {obs_vec[6]:+.3f}, {obs_vec[7]:+.3f}) m",
                f"Camera: OpenCV Generic",
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
            elif key == ord('t'):
                task_start_area_px = ink_obs["ink_area_px"]
                task_started       = True
                print(f"\n[VIS] TASK STARTED. Initial dirt area: {task_start_area_px:.0f} px")
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
    args = parser.parse_args()

    run_vision_loop(half_frame=args.half)
