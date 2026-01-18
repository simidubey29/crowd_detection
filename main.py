import cv2
import numpy as np
import time
import os

# --- PATH SETUP ---
# Adding 'r' before the string tells Python to treat backslashes as literal characters
folder_path = r"C:\Users\panka\PycharmProjects\PythonProject1\data"

# Automatic Selection: Find the first .avi or .mp4 file in that folder
video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4'))]

if not video_files:
    print(f"Error: No video files found in {folder_path}")
    exit()

video_path = os.path.join(folder_path, video_files[0])
print(f"Now playing: {video_files[0]}")

# --- MEMBER 4: REPORT SETUP ---
if not os.path.exists('logs'): os.makedirs('logs')
log_file = open("logs/safety_report.txt", "a")

def log_event(risk, score, msg):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] RISK: {risk} | Score: {score:.2f} | {msg}\n"
    print(entry.strip())
    log_file.write(entry)

# --- PROCESSING ENGINE ---
cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    print("Failed to open the video file.")
    exit()

# Resizing for performance
W, H = 640, 480
prev_gray = cv2.cvtColor(cv2.resize(first_frame, (W, H)), cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_resized = cv2.resize(frame, (W, H))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # MEMBER 2: Motion Engine
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_score = np.mean(mag)

    # MEMBER 3: Risk Logic
    risk, message = "LOW", "Normal Activity"
    if motion_score > 3.0:
        risk, message = "HIGH", "ANOMALY: Sudden Running Detected"
    elif motion_score > 1.2:
        risk, message = "MEDIUM", "CAUTION: Increased Crowd Speed"

    # MEMBER 4: Logging
    if risk != "LOW":
        log_event(risk, motion_score, message)

    # UI Feedback
    color = (0, 0, 255) if risk == "HIGH" else (0, 255, 0)
    cv2.putText(frame_resized, f"RISK: {risk}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Crowd Safety Monitor", frame_resized)

    prev_gray = gray
    if cv2.waitKey(1) & 0xFF == ord('q'): break

log_file.close()
cap.release()
cv2.destroyAllWindows()