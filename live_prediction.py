import os
import json
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

try:
    import keras
    load_model_func = keras.models.load_model
except Exception:
    import tensorflow as tf
    load_model_func = tf.keras.models.load_model

# ============================================================
# SETTINGS & CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join("models", "asl_mobilenetv2_best.keras")
CLASS_NAMES_PATH = os.path.join("models", "class_names.json")

IMAGE_SIZE = 224
MARGIN = 40
CONFIDENCE_THRESHOLD = 0.35
SMOOTHING_WINDOW = 5  # Moving average buffer for prediction stability

# ============================================================
# LOAD CLASS NAMES
# ============================================================

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"class_names.json not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

NUM_CLASSES = len(class_names)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

print("=" * 65)
print("ASL SIGN LANGUAGE RECOGNITION - LIVE PREDICTION SYSTEM")
print("=" * 65)

print("\nLoading MobileNetV2 ASL Model...")
model = load_model_func(MODEL_PATH, compile=False)
print("✅ Model loaded successfully!")
print(f"Model Input Shape  : {model.input_shape}")
print(f"Model Output Shape : {model.output_shape}")
print(f"Total ASL Classes  : {NUM_CLASSES}")

if model.output_shape[-1] != NUM_CLASSES:
    raise ValueError(f"Mismatch between model outputs ({model.output_shape[-1]}) and class_names.json ({NUM_CLASSES})")

# ============================================================
# INITIALIZE MEDIAPIPE HAND TRACKER
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

prob_buffer = deque(maxlen=SMOOTHING_WINDOW)

# ============================================================
# CAMERA INITIALIZATION
# ============================================================

print("\nOpening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Please check camera connection.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened successfully.")
print("\n[INSTRUCTIONS]")
print("1. Show your hand ASL sign clearly to the camera.")
print("2. MediaPipe will isolate your hand and crop the sign.")
print("3. Press 'Q' to quit.\n")

# ============================================================
# REAL-TIME PREDICTION LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    clean_frame = frame.copy()
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
        ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

        x1 = max(0, min(xs) - MARGIN)
        y1 = max(0, min(ys) - MARGIN)
        x2 = min(w, max(xs) + MARGIN)
        y2 = min(h, max(ys) + MARGIN)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        hand_crop = clean_frame[y1:y2, x1:x2]

        if hand_crop.size > 0:
            crop_h, crop_w = hand_crop.shape[:2]

            # 1. Pad to square canvas (Same as collect_data.py)
            side = max(crop_h, crop_w)
            top = (side - crop_h) // 2
            bottom = side - crop_h - top
            left = (side - crop_w) // 2
            right = side - crop_w - left

            square_img = cv2.copyMakeBorder(
                hand_crop, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            # 2. Resize to 224 x 224
            square_img = cv2.resize(square_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

            # 3. Convert BGR to RGB
            input_rgb = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)

            # 4. Convert to float32 in [0..255] (Model handles preprocess_input internally)
            input_tensor = input_rgb.astype(np.float32)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # 5. Model Inference
            raw_probs = model.predict(input_tensor, verbose=0)[0]

            # 6. Smooth predictions over consecutive frames
            prob_buffer.append(raw_probs)
            smoothed_probs = np.mean(prob_buffer, axis=0)

            predicted_idx = int(np.argmax(smoothed_probs))
            predicted_class = class_names[predicted_idx]
            confidence = float(smoothed_probs[predicted_idx])

            top_indices = np.argsort(smoothed_probs)[-3:][::-1]

            # UI Display
            cv2.rectangle(frame, (10, 10), (450, 110), (0, 0, 0), -1)

            if confidence >= CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)
                pred_text = f"ASL Sign: {predicted_class}"
            else:
                color = (0, 255, 255)
                pred_text = f"ASL Sign: {predicted_class} (?)"

            cv2.putText(frame, pred_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.rectangle(frame, (460, 10), (630, 130), (0, 0, 0), -1)
            cv2.putText(frame, "Top 3:", (470, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            for rank, idx in enumerate(top_indices):
                c_name = class_names[idx]
                c_prob = smoothed_probs[idx]
                cv2.putText(frame, f"{rank+1}. {c_name}: {c_prob*100:.1f}%", (470, 60 + rank * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    else:
        prob_buffer.clear()
        cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.putText(frame, "NO HAND DETECTED", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'Q' to quit", (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("ASL Recognition System - Live Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
