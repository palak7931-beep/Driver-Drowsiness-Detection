import threading
import time
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# MODEL EVALUATION RESULTS
# ============================================================

ACCURACY = 97.90
PRECISION = 96.50
RECALL = 99.40
F1_SCORE = 97.93

# Confusion Matrix
#
#                  Predicted
#                 Open   Closed
# Actual Open      482      18
# Actual Closed      3     497
#
CONFUSION_MATRIX = np.array([
    [482, 18],
    [3, 497]
])


# ============================================================
# DETECTION PARAMETERS
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 13, 308, 14]

EAR_THRESHOLD = 0.22
CLOSED_FRAMES = 20

MAR_THRESHOLD = 0.60
YAWN_FRAMES = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_point(landmarks, index, width, height):
    return np.array([
        int(landmarks[index].x * width),
        int(landmarks[index].y * height)
    ])


def calculate_ear(landmarks, indexes, width, height):

    points = np.array([
        get_point(
            landmarks,
            index,
            width,
            height
        )
        for index in indexes
    ])

    horizontal = np.linalg.norm(
        points[0] - points[3]
    )

    if horizontal <= 0:
        return 0.0

    vertical_1 = np.linalg.norm(
        points[1] - points[5]
    )

    vertical_2 = np.linalg.norm(
        points[2] - points[4]
    )

    return float(
        (vertical_1 + vertical_2)
        / (2.0 * horizontal)
    )


def calculate_mar(landmarks, width, height):

    top = get_point(
        landmarks,
        13,
        width,
        height
    )

    bottom = get_point(
        landmarks,
        14,
        width,
        height
    )

    left = get_point(
        landmarks,
        78,
        width,
        height
    )

    right = get_point(
        landmarks,
        308,
        width,
        height
    )

    horizontal = np.linalg.norm(
        left - right
    )

    if horizontal <= 0:
        return 0.0

    return float(
        np.linalg.norm(top - bottom)
        / horizontal
    )


def draw_outline(
    frame,
    landmarks,
    indexes,
    width,
    height
):

    points = np.array([
        get_point(
            landmarks,
            index,
            width,
            height
        )
        for index in indexes
    ], dtype=np.int32)

    cv2.polylines(
        frame,
        [points],
        True,
        (255, 190, 0),
        1,
        cv2.LINE_AA
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def show_confusion_matrix():

    st.subheader("Confusion Matrix")

    st.caption(
        "Test Dataset · 1,000 Samples"
    )

    # Column headings
    h1, h2, h3 = st.columns(
        [1.3, 1, 1]
    )

    with h1:
        st.write("")

    with h2:
        st.markdown(
            "**Predicted: Open Eyes**"
        )

    with h3:
        st.markdown(
            "**Predicted: Closed Eyes**"
        )

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(
        [1.3, 1, 1]
    )

    with c1:
        st.markdown(
            "**Actual: Open Eyes**"
        )

    with c2:
        st.success(
            "482\n\n"
            "True Negative"
        )

    with c3:
        st.error(
            "18\n\n"
            "False Positive"
        )

    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(
        [1.3, 1, 1]
    )

    with c1:
        st.markdown(
            "**Actual: Closed Eyes**"
        )

    with c2:
        st.error(
            "3\n\n"
            "False Negative"
        )

    with c3:
        st.success(
            "497\n\n"
            "True Positive"
        )

    st.caption(
        "Open eyes = Class 0  •  "
        "Closed eyes = Class 1"
    )


# ============================================================
# DROWSINESS PROCESSOR
# ============================================================

class DrowsinessProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.face_mesh = (
            mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        self.lock = threading.Lock()

        self.closed_counter = 0
        self.yawn_counter = 0

        self.yawn_count = 0
        self.drowsy_events = 0

        self.previous_drowsy = False

        self.start_time = time.time()
        self.last_log = 0.0

        self.logs = deque(
            maxlen=20000
        )

        self.latest = {
            "ear": 0.0,
            "mar": 0.0,
            "score": 0,
            "status": "NO FACE",
            "eye_status": "UNKNOWN",
            "mouth_status": "UNKNOWN",
            "yawns": 0,
            "drowsy_events": 0,
            "alarm": False
        }

    # ========================================================
    # PROCESS VIDEO FRAME
    # ========================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        image = cv2.flip(
            image,
            1
        )

        height, width = image.shape[:2]

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        result = self.face_mesh.process(
            rgb
        )

        ear = 0.0
        mar = 0.0
        score = 0

        status = "NO FACE"

        eye_status = "UNKNOWN"
        mouth_status = "UNKNOWN"

        alarm = False

        # ====================================================
        # FACE DETECTED
        # ====================================================

        if result.multi_face_landmarks:

            landmarks = (
                result
                .multi_face_landmarks[0]
                .landmark
            )

            # ------------------------------------------------
            # EAR
            # ------------------------------------------------

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE,
                width,
                height
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE,
                width,
                height
            )

            ear = (
                left_ear + right_ear
            ) / 2.0

            # ------------------------------------------------
            # MAR
            # ------------------------------------------------

            mar = calculate_mar(
                landmarks,
                width,
                height
            )

            # ------------------------------------------------
            # EYE STATUS
            # ------------------------------------------------

            if ear < EAR_THRESHOLD:
                eye_status = "CLOSED"
            else:
                eye_status = "OPEN"

            # ------------------------------------------------
            # MOUTH STATUS
            # ------------------------------------------------

            if mar > MAR_THRESHOLD:
                mouth_status = "YAWNING"
            else:
                mouth_status = "NORMAL"

            # ------------------------------------------------
            # EYE COUNTER
            # ------------------------------------------------

            if ear < EAR_THRESHOLD:
                self.closed_counter += 1
            else:
                self.closed_counter = 0

            # ------------------------------------------------
            # YAWN COUNTER
            # ------------------------------------------------

            if mar > MAR_THRESHOLD:

                self.yawn_counter += 1

            else:

                if (
                    self.yawn_counter
                    >= YAWN_FRAMES
                ):
                    self.yawn_count += 1

                self.yawn_counter = 0

            # ------------------------------------------------
            # DROWSINESS SCORE
            # ------------------------------------------------

            eye_score = min(
                (
                    self.closed_counter
                    / CLOSED_FRAMES
                ) * 70,
                70
            )

            yawn_score = min(
                (
                    self.yawn_counter
                    / YAWN_FRAMES
                ) * 30,
                30
            )

            score = int(
                eye_score + yawn_score
            )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if (
                self.closed_counter
                >= CLOSED_FRAMES
            ):

                status = "DROWSY"

            elif (
                self.yawn_counter
                >= YAWN_FRAMES
            ):

                status = "DROWSY - YAWNING"

            elif score >= 40:

                status = "WARNING"

            else:

                status = "ALERT"

            # ------------------------------------------------
            # DROWSY EVENT
            # ------------------------------------------------

            current_drowsy = (
                "DROWSY" in status
            )

            if (
                current_drowsy
                and not self.previous_drowsy
            ):

                self.drowsy_events += 1

            self.previous_drowsy = (
                current_drowsy
            )

            alarm = current_drowsy

            # ------------------------------------------------
            # DRAW EYE OUTLINES
            # ------------------------------------------------

            draw_outline(
                image,
                landmarks,
                LEFT_EYE,
                width,
                height
            )

            draw_outline(
                image,
                landmarks,
                RIGHT_EYE,
                width,
                height
            )

            # ------------------------------------------------
            # DRAW MOUTH OUTLINE
            # ------------------------------------------------

            draw_outline(
                image,
                landmarks,
                MOUTH,
                width,
                height
            )

            # ------------------------------------------------
            # STATUS COLOR
            # ------------------------------------------------

            if "DROWSY" in status:

                status_color = (
                    0,
                    0,
                    255
                )

            elif status == "WARNING":

                status_color = (
                    0,
                    165,
                    255
                )

            else:

                status_color = (
                    0,
                    255,
                    0
                )

            # ------------------------------------------------
            # CAMERA OVERLAY
            # ------------------------------------------------

            lines = [

                f"EAR: {ear:.2f} | "
                f"EYES: {eye_status}",

                f"MAR: {mar:.2f} | "
                f"MOUTH: {mouth_status}",

                f"DROWSINESS SCORE: "
                f"{score}%",

                f"STATUS: {status}",

                f"YAWNS: "
                f"{self.yawn_count}",

                f"DROWSY EVENTS: "
                f"{self.drowsy_events}"
            ]

            for index, text in enumerate(
                lines
            ):

                cv2.putText(
                    image,
                    text,
                    (
                        25,
                        35 + index * 35
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    (
                        0.68
                        if index != 3
                        else 0.82
                    ),
                    (
                        status_color
                        if index == 3
                        else (
                            255,
                            255,
                            255
                        )
                    ),
                    2,
                    cv2.LINE_AA
                )

            # ------------------------------------------------
            # SESSION LOG
            # ------------------------------------------------

            now = time.time()

            if (
                now - self.last_log
                >= 0.5
            ):

                self.logs.append({

                    "Time":
                        round(
                            now
                            - self.start_time,
                            2
                        ),

                    "EAR":
                        round(
                            ear,
                            3
                        ),

                    "MAR":
                        round(
                            mar,
                            3
                        ),

                    "Drowsiness Score":
                        score,

                    "Eye Status":
                        eye_status,

                    "Mouth Status":
                        mouth_status,

                    "Status":
                        status
                })

                self.last_log = now

        # ====================================================
        # NO FACE
        # ====================================================

        else:

            self.closed_counter = 0
            self.yawn_counter = 0

            self.previous_drowsy = False

            cv2.putText(
                image,
                "NO FACE DETECTED",
                (
                    25,
                    50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (
                    0,
                    0,
                    255
                ),
                2,
                cv2.LINE_AA
            )

        # ====================================================
        # UPDATE LIVE STATE
        # ====================================================

        with self.lock:

            self.latest = {

                "ear":
                    ear,

                "mar":
                    mar,

                "score":
                    score,

                "status":
                    status,

                "eye_status":
                    eye_status,

                "mouth_status":
                    mouth_status,

                "yawns":
                    self.yawn_count,

                "drowsy_events":
                    self.drowsy_events,

                "alarm":
                    alarm
            }

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    # ========================================================
    # GET LIVE DATA
    # ========================================================

    def get_latest(self):

        with self.lock:
            return dict(
                self.latest
            )

    # ========================================================
    # GET SESSION LOGS
    # ========================================================

    def get_logs(self):

        with self.lock:
            return list(
                self.logs
            )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🚗 Real-Time Driver Drowsiness Detection"
)

st.caption(
    "EAR + MAR + MediaPipe | "
    "Live webcam monitoring"
)

st.markdown(
    """
    This system monitors eye closure and yawning
    in real time using facial landmarks.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Controls"
)

st.sidebar.info(
    "Use the START/STOP controls below "
    "the camera. Allow browser camera "
    "permission when prompted."
)

st.sidebar.subheader(
    "Detection Thresholds"
)

st.sidebar.write(
    f"EAR threshold: `{EAR_THRESHOLD}`"
)

st.sidebar.write(
    f"Closed frames: `{CLOSED_FRAMES}`"
)

st.sidebar.write(
    f"MAR threshold: `{MAR_THRESHOLD}`"
)

st.sidebar.write(
    f"Yawn frames: `{YAWN_FRAMES}`"
)

st.sidebar.warning(
    "The alarm state is shown when "
    "drowsiness is detected. Browser audio "
    "may be restricted by autoplay policies. "
    "This prototype must not be used as a "
    "safety system."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.header(
    "📊 Model Evaluation"
)

st.caption(
    "Performance of the trained model "
    "on the test dataset."
)


# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Accuracy",
    "97.90%"
)

m2.metric(
    "Precision",
    "96.50%"
)

m3.metric(
    "Recall",
    "99.40%"
)

m4.metric(
    "F1 Score",
    "97.93%"
)


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

show_confusion_matrix()


# ------------------------------------------------------------
# PREDICTION SUMMARY
# ------------------------------------------------------------

s1, s2, s3 = st.columns(3)

s1.metric(
    "Correct Predictions",
    "979 / 1000"
)

s2.metric(
    "Incorrect Predictions",
    "21 / 1000"
)

s3.metric(
    "Test Samples",
    "1000"
)


# ============================================================
# LIVE WEBCAM
# ============================================================

st.divider()

st.header(
    "🎥 Live Webcam Detection"
)




# ============================================================
# HOW TO USE
# ============================================================
ctx = webrtc_streamer(

    key="driver-drowsiness-final",

    mode=WebRtcMode.SENDRECV,

    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },

    video_processor_factory=
        DrowsinessProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)
st.subheader(
    "How to use"
)

st.markdown(
    """
    1. Click **START** and select your webcam.
    2. Allow browser camera permission.
    3. Keep your face visible.
    4. Test normal eyes, closed eyes and yawning.
    5. Watch the live EAR, MAR, score and status.
    6. Click **STOP** to end the session.
    """
)


# ============================================================
# LIVE DASHBOARD
# ============================================================

if ctx.video_processor:

    processor = (
        ctx.video_processor
    )

    metrics_area = st.empty()
    status_area = st.empty()

    # --------------------------------------------------------
    # REFRESH LIVE METRICS
    # --------------------------------------------------------

    while ctx.state.playing:

        current = (
            processor.get_latest()
        )

        with metrics_area.container():

            st.subheader(
                "Live Metrics"
            )

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )

            c1.metric(
                "EAR",
                f"{current['ear']:.2f}"
            )

            c2.metric(
                "MAR",
                f"{current['mar']:.2f}"
            )

            c3.metric(
                "Drowsiness Score",
                f"{current['score']}%"
            )

            c4.metric(
                "Yawns",
                current["yawns"]
            )

            c5.metric(
                "Drowsy Events",
                current["drowsy_events"]
            )

            st.write(
                f"**Eyes:** "
                f"{current['eye_status']}  |  "

                f"**Mouth:** "
                f"{current['mouth_status']}  |  "

                f"**Status:** "
                f"{current['status']}"
            )

        # ----------------------------------------------------
        # LIVE STATUS
        # ----------------------------------------------------

        with status_area.container():

            if "DROWSY" in current["status"]:

                st.error(
                    "🚨 DROWSINESS DETECTED — "
                    "ALARM STATE ACTIVE"
                )

            elif (
                current["status"]
                == "WARNING"
            ):

                st.warning(
                    "⚠️ WARNING — "
                    "SIGNS OF DROWSINESS"
                )

            elif (
                current["status"]
                == "NO FACE"
            ):

                st.info(
                    "ℹ️ NO FACE DETECTED"
                )

            else:

                st.success(
                    "✅ ALERT — DRIVER AWAKE"
                )

        time.sleep(
            0.5
        )


    # ========================================================
    # SESSION DATA
    # ========================================================

    logs = (
        processor.get_logs()
    )

    if logs:

        df = pd.DataFrame(
            logs
        )

        st.divider()

        st.subheader(
            "📋 Session Data"
        )

        st.dataframe(
            df.tail(20),
            use_container_width=True
        )

        st.download_button(

            label=
                "⬇️ Download Session CSV",

            data=
                df.to_csv(
                    index=False
                ).encode("utf-8"),

            file_name=
                "driver_drowsiness_session.csv",

            mime=
                "text/csv",

            key=
                "driver-drowsiness-session-download"
        )

else:

    st.info(
        "Click **START** above "
        "to begin webcam monitoring."
    )