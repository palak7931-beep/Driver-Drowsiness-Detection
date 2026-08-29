import cv2
import mediapipe as mp
import numpy as np
import pygame
import csv
import os
import time

# ==========================================
# ALARM SETUP
# ==========================================

pygame.mixer.init()

sample_rate = 44100
duration = 0.5
frequency = 1000

t = np.linspace(
    0,
    duration,
    int(sample_rate * duration),
    False
)

tone = np.sin(2 * np.pi * frequency * t)

audio = (tone * 32767).astype(np.int16)

stereo_audio = np.column_stack((audio, audio))

alarm_sound = pygame.sndarray.make_sound(stereo_audio)


# ==========================================
# MEDIAPIPE
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# LANDMARKS
# ==========================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ==========================================
# PARAMETERS
# ==========================================

EAR_THRESHOLD = 0.22
CLOSED_FRAMES = 20

MAR_THRESHOLD = 0.60
YAWN_FRAMES = 10

closed_counter = 0
yawn_counter = 0

yawn_count = 0
drowsy_events = 0

previous_drowsy = False


# ==========================================
# EAR CALCULATION
# ==========================================

def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for point in eye_points:

        x = int(landmarks[point].x * width)
        y = int(landmarks[point].y * height)

        points.append([x, y])

    points = np.array(points)

    vertical_1 = np.linalg.norm(
        points[1] - points[5]
    )

    vertical_2 = np.linalg.norm(
        points[2] - points[4]
    )

    horizontal = np.linalg.norm(
        points[0] - points[3]
    )

    return (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)


# ==========================================
# MAR CALCULATION
# ==========================================

def calculate_mar(landmarks, width, height):

    top = landmarks[13]
    bottom = landmarks[14]

    left = landmarks[78]
    right = landmarks[308]

    top_point = np.array([
        int(top.x * width),
        int(top.y * height)
    ])

    bottom_point = np.array([
        int(bottom.x * width),
        int(bottom.y * height)
    ])

    left_point = np.array([
        int(left.x * width),
        int(left.y * height)
    ])

    right_point = np.array([
        int(right.x * width),
        int(right.y * height)
    ])

    vertical = np.linalg.norm(
        top_point - bottom_point
    )

    horizontal = np.linalg.norm(
        left_point - right_point
    )

    return vertical / horizontal


# ==========================================
# RESULTS CSV
# ==========================================

os.makedirs("results", exist_ok=True)

csv_path = "results/final_session.csv"

csv_file = open(
    csv_path,
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Time",
    "EAR",
    "MAR",
    "Drowsiness Score",
    "Status"
])


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

start_time = time.time()

last_log = 0


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb)

    status = "NO FACE"

    ear = 0
    mar = 0
    score = 0


    # ======================================
    # FACE DETECTED
    # ======================================

    if results.multi_face_landmarks:

        landmarks = (
            results.multi_face_landmarks[0].landmark
        )


        # EAR

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
        ) / 2


        # MAR

        mar = calculate_mar(
            landmarks,
            width,
            height
        )


        # ==================================
        # EYE CLOSURE
        # ==================================

        if ear < EAR_THRESHOLD:

            closed_counter += 1

        else:

            closed_counter = 0


        # ==================================
        # YAWN
        # ==================================

        if mar > MAR_THRESHOLD:

            yawn_counter += 1

        else:

            if yawn_counter >= YAWN_FRAMES:

                yawn_count += 1

            yawn_counter = 0


        # ==================================
        # DROWSINESS SCORE
        # ==================================

        eye_score = min(
            (closed_counter / CLOSED_FRAMES) * 70,
            70
        )

        yawn_score = min(
            (yawn_counter / YAWN_FRAMES) * 30,
            30
        )

        score = int(
            eye_score + yawn_score
        )


        # ==================================
        # STATUS
        # ==================================

        if closed_counter >= CLOSED_FRAMES:

            status = "DROWSY"

        elif yawn_counter >= YAWN_FRAMES:

            status = "DROWSY - YAWNING"

        elif score >= 40:

            status = "WARNING"

        else:

            status = "ALERT"


        # ==================================
        # COUNT EVENTS
        # ==================================

        current_drowsy = (
            "DROWSY" in status
        )

        if current_drowsy and not previous_drowsy:

            drowsy_events += 1

        previous_drowsy = current_drowsy


        # ==================================
        # ALARM
        # ==================================

        if current_drowsy:

            if not pygame.mixer.get_busy():

                alarm_sound.play(-1)

        else:

            alarm_sound.stop()


        # ==================================
        # DISPLAY
        # ==================================

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drowsiness Score: {score}%",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # Status color

        if "DROWSY" in status:

            status_color = (0, 0, 255)

        elif status == "WARNING":

            status_color = (0, 165, 255)

        else:

            status_color = (0, 255, 0)


        cv2.putText(
            frame,
            f"STATUS: {status}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            3
        )

        cv2.putText(
            frame,
            f"Yawns: {yawn_count}",
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drowsy Events: {drowsy_events}",
            (30, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ==================================
        # LOG DATA
        # ==================================

        current_time = time.time()

        if current_time - last_log >= 0.5:

            writer.writerow([
                round(
                    current_time - start_time,
                    2
                ),
                round(ear, 3),
                round(mar, 3),
                score,
                status
            ])

            csv_file.flush()

            last_log = current_time


    else:

        closed_counter = 0
        yawn_counter = 0

        previous_drowsy = False

        alarm_sound.stop()

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ======================================
    # SHOW
    # ======================================

    cv2.imshow(
        "Real-Time Driver Drowsiness Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

alarm_sound.stop()

cap.release()

cv2.destroyAllWindows()

face_mesh.close()

pygame.quit()

csv_file.close()


print("\n================================")
print("SESSION COMPLETED")
print("================================")
print(f"Yawns detected: {yawn_count}")
print(f"Drowsy events: {drowsy_events}")
print(f"Session data: {csv_path}")
print("================================")