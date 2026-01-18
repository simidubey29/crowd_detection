import streamlit as st
import cv2
import numpy as np
import tempfile
import time

# =========================
# FINAL THRESHOLDS (FROM CALIBRATION)
# =========================
VERY_LOW   = 0.60
NORMAL     = 1.20
ELEVATED   = 2.50
HIGH       = 4.00
CRITICAL   = 6.00

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(page_title="Crowd Risk Detection", layout="wide")

st.title("🚨 Crowd Behavior Risk Detection System")
st.markdown("""
*Technique:* Optical Flow (OpenCV)  
*Focus:* Crowd-level motion only  
*Ethical:* No face recognition, no tracking
""")

source = st.radio("Select Input Source:", ["Upload Video", "Use Webcam"])

video_box = st.empty()
alert_box = st.empty()
result_box = st.empty()   # 🔹 FINAL RESULT BOX

# =========================
# INPUT SOURCE
# =========================
cap = None

if source == "Upload Video":
    uploaded_video = st.file_uploader(
        "Upload CCTV / Crowd Video",
        type=["mp4", "avi", "mov"]
    )
    if uploaded_video:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_video.read())
        cap = cv2.VideoCapture(temp.name)
else:
    cap = cv2.VideoCapture(0)

# =========================
# START BUTTON
# =========================
if st.button("▶ Start Detection") and cap is not None:

    ret, prev_frame = cap.read()
    if not ret:
        st.error("Unable to read video")
        st.stop()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    prev_motion = 0
    alerts = []

    # 🔹 FINAL RESULT STORAGE
    motion_history = []
    risk_counter = {
        "VERY LOW": 0,
        "NORMAL": 0,
        "ELEVATED": 0,
        "HIGH RISK": 0,
        "CRITICAL": 0
    }

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_motion = np.mean(mag)
        motion_history.append(avg_motion)

        # =========================
        # MOTION SPIKE DETECTION
        # =========================
        spike = avg_motion - prev_motion

        # =========================
        # RISK CLASSIFICATION
        # =========================
        if avg_motion < VERY_LOW:
            risk = "VERY LOW"
            color = (200, 200, 200)
            reason = "Minimal movement"

        elif avg_motion < NORMAL:
            risk = "NORMAL"
            color = (0, 255, 0)
            reason = "Stable crowd flow"

        elif avg_motion < ELEVATED:
            risk = "ELEVATED"
            color = (0, 255, 255)
            reason = "Increased activity"

        elif avg_motion < HIGH:
            risk = "HIGH RISK"
            color = (0, 165, 255)
            reason = "Abnormal acceleration"

        else:
            risk = "CRITICAL"
            color = (0, 0, 255)
            reason = "Possible panic or stampede"

        if spike > 2.0:
            risk = "CRITICAL"
            reason = "Sudden motion spike detected"

        risk_counter[risk] += 1

        # =========================
        # DISPLAY ON FRAME
        # =========================
        cv2.putText(frame, f"Risk: {risk}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Motion: {avg_motion:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, reason, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_box.image(frame_rgb, channels="RGB")

        # =========================
        # ALERT LOG
        # =========================
        if risk in ["HIGH RISK", "CRITICAL"]:
            timestamp = time.strftime("%H:%M:%S")
            alerts.append(f"{timestamp} | {risk} | {reason}")
            alert_box.warning("\n".join(alerts[-5:]))

        prev_gray = gray
        prev_motion = avg_motion
        frame_count += 1

    cap.release()

    # =========================
    # FINAL RESULTS SECTION ⭐
    # =========================
    avg_video_motion = np.mean(motion_history)
    dominant_risk = max(risk_counter, key=risk_counter.get)

    if dominant_risk in ["VERY LOW", "NORMAL"]:
        verdict = "🟢 SAFE CROWD"
    elif dominant_risk == "ELEVATED":
        verdict = "🟡 CROWD NEEDS MONITORING"
    else:
        verdict = "🔴 DANGEROUS CROWD CONDITION"

    fps = frame_count / (time.time() - start_time)

    result_box.success(f"""
### 📊 FINAL ANALYSIS REPORT

* Total Frames Analyzed: *{frame_count}*  
* Average Crowd Motion: *{avg_video_motion:.2f}*  
* Dominant Risk Level: *{dominant_risk}*  
* Final Verdict: *{verdict}*  
* Processing Speed: *{fps:.2f} FPS*

#### Risk Distribution:
- VERY LOW: {risk_counter["VERY LOW"]}
- NORMAL: {risk_counter["NORMAL"]}
- ELEVATED: {risk_counter["ELEVATED"]}
- HIGH RISK: {risk_counter["HIGH RISK"]}
- CRITICAL: {risk_counter["CRITICAL"]}
""")

else:
    st.info("Upload a video or select webcam, then click Start Detection")