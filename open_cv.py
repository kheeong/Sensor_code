import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from picamera2 import Picamera2
import serial
# -----------------------------
# Settings
# -----------------------------
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
# -----------------------------
while True:
    frame = picam2.capture_array()
    ret = True
    if not ret:   # video ended
        break

    centroids, mask = find_green_centroids(frame)

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

        angle_history.append(relative_angle)
        v_raw = model.update(relative_angle)
        v_f   = ekf.update(v_raw)
        uart.write(f"{v_f:.2f}\n".encode())
        uart.flush()
        # Draw
        #cv2.circle(frame, (x1, y1), 8, (0, 0, 255), -1)
        #cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)
        #cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        #cv2.putText(frame, f"Angle: {relative_angle:.2f} deg",(30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Update plot
    line.set_xdata(range(len(angle_history)))
    line.set_ydata(angle_history)
    ax.set_xlim(0, max(200, len(angle_history)))
    fig.canvas.draw()
    fig.canvas.flush_events()

    #cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup OpenCV (BUT keep plot open)
# -----------------------------
#cap.release()

import csv

with open("angle_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "angle_deg"])
    for i, angle in enumerate(angle_history):
        writer.writerow([i, angle])

print("Saved CSV: angle_history.csv")

cv2.destroyAllWindows()

# Keep the plot open
plt.ioff()
plt.show()
