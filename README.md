Crowd Behavior Risk Detection System

<img width="1920" height="1080" alt="Screenshot (125)" src="https://github.com/user-attachments/assets/a51ad3d9-40ae-46a6-b7e4-42d1944a2da1" />


PROJECT OVERVIEW:

This project is a logic-driven AI system designed to detect abnormal crowd behavior in real-time or from pre-recorded videos. It focuses strictly on crowd-level patterns, avoiding individual identification.
The system is beginner-friendly and uses optical flow, density estimation, and rule-based risk thresholds to detect potential panic, stampedes, or unsafe collective movements.

HACKATHON PROBLEM STATEMENT:

In crowded public areas like markets, metro stations, and festivals, safety incidents such as panic, stampedes, or sudden crowd movements can escalate quickly. Traditional CCTV systems rely on human monitoring, which is:
- Slow to respond
- Prone to error
- Difficult to scale

Task:
Build a system that automatically analyzes crowd behavior and identifies public safety risks in real-time.

Objectives:
- Detect unusual or unsafe crowd behavior
- Identify collective movement patterns
- Classify detected events into risk levels
- Generate interpretable alerts

Core Constraints:
- Only crowd-level analysis (no face recognition or tracking)
- Use video input (live feed optional)
- Explainable, ethical AI

FEATURES:

- Real-time crowd motion analysis
- Panic spike detection
- Zone-based risk assessment
- Stampede prediction using motion trends
- Dynamic risk thresholds with calibration
- Detailed logging & console output
- Streamlit interactive dashboard
- Final summary report with stability score and risk verdict

SYSTEM DESIGN

<img width="1910" height="312" alt="Screenshot (125)c" src="https://github.com/user-attachments/assets/5523409f-84a8-46b2-bd20-434ae3e453c9" />

Input Sources:
- Upload video (.mp4, .avi)
- Webcam (local only)

Preprocessing:
- Convert frames to grayscale
- Apply background subtraction for density
- Compute optical flow per zone

Motion Analysis:
- Average motion per frame
- Smooth motion using a buffer
- Detect sudden spikes (panic events)

Calibration:
- Collect motion data from normal videos
- Compute median, MAD, mean, std
- Define thresholds: VERY_LOW, NORMAL, ELEVATED, HIGH, CRITICAL

Risk Detection:
- Assign risk level per frame
- Track consecutive critical frames for emergency alerts
- Predict stampede based on slope of motion trend

Output & Visualization:
- Streamlit dashboard metrics
- Dynamic plots of motion intensity
- Console logs & ASCII charts for monitoring
<img width="1920" height="1080" alt="Screenshot (127)" src="https://github.com/user-attachments/assets/378aae95-3ee0-4358-ac72-6e95737c9280" />

  
- Emergency alerts with human-readable explanation
  

<img width="1920" height="1080" alt="Screenshot (126)" src="https://github.com/user-attachments/assets/ba9a3f58-44ac-45f4-a7ff-9afa676308e4" />


TECHNICAL DETAILS

Languages & Libraries:
- Python 3.x
- OpenCV (cv2)
- NumPy (numpy)
- Streamlit (streamlit)
- Matplotlib (matplotlib)
- Logging & system libraries

Algorithms Used:
- Optical Flow (Farneback) for motion tracking
- Background subtraction for density estimation

DATASET:

The dataset consists of 5 pre-recorded crowd videos used for calibration and testing:

-Video Name      Type         Description
-normal1.mp4     Calibration  Normal crowd in a market
-normal2.mp4     Calibration  Stable crowd in a metro station
-normal3.mp4     Calibration  Regular festival crowd
-risk1.mp4       Test / Risk  Simulated crowd panic scenario
-risk2.mp4       Test / Risk  High-density crowd with sudden motion

Dataset Guidelines:
- Videos are either public, self-recorded, or synthetic
- Only crowd-level behavior analyzed
- Preprocessing includes resizing and grayscale conversion

INSTALLATION & USAGE
1. Clone the repository:
   git clone <repository-url>
   cd crowd-risk-detection

2. Install dependencies:
   pip install -r requirements.txt

3. Run the app:
   streamlit run app.py

4. Upload a video or select webcam to start detection.

CALIBRATION

<img width="1920" height="1080" alt="Screenshot (128)" src="https://github.com/user-attachments/assets/412f99df-5ea4-44d2-ba54-92c27f6b7167" />

- Calibration allows the system to set dynamic thresholds based on normal crowd behavior.
- Process multiple “normal” videos.
- Compute median, MAD, mean, max of motion per video.
- Derive thresholds: VERY_LOW, NORMAL, ELEVATED, HIGH, CRITICAL, SPIKE_THRESHOLD
- The calibration script is included as calibration.py to generate these values automatically.

OUTPUTS & DASHBOARD:

<img width="1920" height="1080" alt="Screenshot (129)" src="https://github.com/user-attachments/assets/e489d3d7-250b-4085-8254-e10be0a06a26" />

Dashboard Metrics:
- Frames processed
- Average motion
- Risky zones
- Panic spikes
- High alerts
- Stampede risk

Final Summary:
- Total frames & FPS
- Average motion confidence
- Crowd stability score
- Risk distribution
- Final verdict: 
  🟢 SAFE CROWD
  🟠 UNSTABLE – INTERVENTION ADVISED
  🔴 STAMPEDE HIGHLY LIKELY
<img width="1920" height="1080" alt="Screenshot (130)" src="https://github.com/user-attachments/assets/b0249b49-021a-4b65-a024-4f6b58e8435d" />

Emergency Alerts:
- Triggered for consecutive critical frames
- Includes human-readable explanation & recommended actions

Visualization:
- Motion intensity plot over time
- ASCII console charts for motion & risk trends

ETHICAL CONSIDERATIONS:

- No face recognition or individual tracking
- Only crowd-level patterns are analyzed
- Explainable alerts only
- Compliant with privacy and ethical AI guidelines

FUTURE IMPROVEMENTS:

- Add live multi-camera integration
- Use machine learning models for better anomaly detection
- Integrate real-time alert notifications (SMS/email)
- Enhance zone segmentation for more granular risk detection
- Include historical trend analysis for predictive safety

REFERENCES:
- OpenCV Optical Flow Documentation: https://docs.opencv.org/
- Streamlit Docs: https://docs.streamlit.io/
- Hackathon Problem Statement
- Statistical thresholds (median, MAD, std)
- Rule-based spike and risk detection
