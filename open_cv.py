import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from picamera2 import Picamera2
import serial
import threading
# -----------------------------
# Settings
# -----------------------------
MEAS_HZ = 8.0          # vision measurement rate (sample)
OUT_HZ  = 200.0        # output/logging rate (hold)
MEAS_DT = 1.0 / MEAS_HZ
OUT_DT  = 1.0 / OUT_HZ

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
picam2.configure(config)
picam2.start()
uart = serial.Serial(
    port="/dev/ttyAMA0",
    baudrate=115200,
    timeout=1,
    write_timeout=2
)
time.sleep(0.3)


USE_WEBCAM = False
VIDEO_PATH = "test_video.mp4"

LOWER_GREEN = np.array([35, 70, 70])
UPPER_GREEN = np.array([85, 255, 255])

# Store angle history
angle_history = deque(maxlen=2000)

# -----------------------------
def find_green_centroids(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask = cv2.medianBlur(mask, 7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroids.append((cx, cy))

    return centroids, mask

# -----------------------------
#cap = cv2.VideoCapture(0) if USE_WEBCAM else cv2.VideoCapture(VIDEO_PATH)

calibrated = False
baseline_angle = 0

# Setup plot
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], linewidth=2)
ax.set_ylim(-90, 90)
ax.set_xlim(0, 2000)
ax.set_xlabel("Frame")
ax.set_ylabel("Angle (deg)")
ax.set_title("Pendulum Angle")


class WindModel:
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def update(self, theta_rad):
        # Clip to avoid tan(90°)
        th = np.clip(theta_rad, -np.deg2rad(85), np.deg2rad(85))

        # compute terms
        tan_th = np.tan(th)
        tan_th = max(0.0, tan_th)     # avoid negative domain issues

        th_pos = max(0.0, theta_rad)  # ensure non-negative for theta^d

        # compute model
        v = self.a * (tan_th ** self.b) + self.c * (th_pos ** self.d)
        return v

class EKF1D:
    def __init__(self, q, r):
        self.q = q      # process noise
        self.r = r      # measurement noise
        self.x = 0.0    # state estimate
        self.p = 1.0    # covariance
        self.initialized = False

    def update(self, z):
        # first measurement initializes the filter
        if not self.initialized:
            self.x = z
            self.initialized = True

        # Predict
        x_pred = self.x
        p_pred = self.p + self.q

        # Update
        K = p_pred / (p_pred + self.r)
        self.x = x_pred + K * (z - x_pred)
        self.p = (1 - K) * p_pred

        return self.x

ekf = EKF1D(q=0.021219, r=0.414340)
model = WindModel(a=4.9976, b=0.6871, c=-2.4326, d=1.8387)

lock = threading.Lock()
last_v = 0.0
last_mask = None
have_sample = False

calibrated = False
baseline_angle = 0.0

stop_flag = False

def measurement_thread():
    """Runs at ~8 Hz: capture frame -> compute new sample -> publish last_v."""
    global last_v, last_mask, have_sample, calibrated, baseline_angle, stop_flag

    next_t = time.monotonic()
    while not stop_flag:
        now = time.monotonic()
        if now < next_t:
            time.sleep(next_t - now)
        next_t += MEAS_DT

        frame = picam2.capture_array()

        centroids, mask = find_green_centroids(frame)

        v_f = None
        if len(centroids) == 2:
            centroids = sorted(centroids, key=lambda x: x[1])
            (x1, y1), (x2, y2) = centroids
            dx = x2 - x1
            dy = y2 - y1
            angle_deg = math.degrees(math.atan2(dx, dy))

            if not calibrated:
                baseline_angle = angle_deg
                calibrated = True

            relative_angle = angle_deg - baseline_angle

            v_raw = model.update(np.deg2rad(relative_angle))
            v_f = ekf.update(v_raw)

        with lock:
            last_mask = mask
            if v_f is not None:
                last_v = float(v_f)
                have_sample = True
            # if v_f is None: do nothing -> HOLD last_v

# -----------------------------
# Plot (update less often)
# -----------------------------
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], linewidth=2)
ax.set_ylim(-90, 90)
ax.set_xlabel("Sample @200Hz")
ax.set_ylabel("v_f")
ax.set_title("Held / Upsampled Output (200 Hz)")

# Start measurement sampler
t = threading.Thread(target=measurement_thread, daemon=True)
t.start()

# -----------------------------
# Output loop at 200 Hz (sample-and-hold)
# -----------------------------
next_out = time.monotonic()
plot_div = 10  # update plot every 10 outputs = 20 Hz plotting (lighter)
k = 0

try:
    while True:
        now = time.monotonic()
        if now < next_out:
            time.sleep(next_out - now)
        next_out += OUT_DT

        with lock:
            v_out = last_v
            mask = last_mask
            ok = have_sample

        if ok:
            angle_history.append(v_out)
            uart.write(f"{v_out:.2f}\n".encode())
            # uart.flush()  # usually not needed every line; uncomment if required

        # Plot less frequently (don’t try to redraw at 200 Hz)
        if (k % plot_div) == 0 and len(angle_history) > 2:
            line.set_xdata(range(len(angle_history)))
            line.set_ydata(angle_history)
            ax.set_xlim(max(0, len(angle_history) - int(OUT_HZ * 5)), len(angle_history))  # last 5s
            fig.canvas.draw()
            fig.canvas.flush_events()

        # Show mask (8 Hz updates; will "hold" between samples too)
        if mask is not None:
            cv2.imshow("mask", mask)

        k += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    stop_flag = True
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

    # Save CSV (200 Hz samples)
    import csv
    with open("angle_history_200hz.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx_200hz", "v_f_held"])
        for i, v in enumerate(angle_history):
            w.writerow([i, v])
    print("Saved CSV: angle_history_200hz.csv")