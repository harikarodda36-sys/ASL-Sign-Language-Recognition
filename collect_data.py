import cv2
import mediapipe as mp
import os
import time

# ============================================================
# SETTINGS
# ============================================================

LETTER = "OK"

# FIRST TEST: collect only 10 images
TARGET_IMAGES = 300

# NEW DATASET - OLD DATASET WILL NOT BE TOUCHED
SAVE_FOLDER = os.path.join("new_dataset", LETTER)

IMAGE_SIZE = 224
MARGIN = 40
CAPTURE_INTERVAL = 0.20

os.makedirs(SAVE_FOLDER, exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# ============================================================
# CHECK EXISTING NEW IMAGES
# ============================================================

existing_images = [
    f for f in os.listdir(SAVE_FOLDER)
    if f.lower().endswith(".jpg")
]

count = len(existing_images)


print("--------------------------------------")
print("ASL CLEAN IMAGE COLLECTION")
print("--------------------------------------")
print("Letter:", LETTER)
print("Target:", TARGET_IMAGES)
print("Already collected:", count)
print()
print(f"Make the ASL letter {LETTER}.")
print("SPACE = Start collecting")
print("Q = Quit")
print("--------------------------------------")


started = False
last_save_time = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # --------------------------------------------------------
    # IMPORTANT:
    # Save this clean frame BEFORE drawing anything.
    # --------------------------------------------------------

    clean_frame = frame.copy()

    # --------------------------------------------------------
    # MEDIAPIPE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        h, w, _ = clean_frame.shape

        # ----------------------------------------------------
        # FIND HAND BOUNDING BOX
        # ----------------------------------------------------

        xs = [
            int(landmark.x * w)
            for landmark in hand_landmarks.landmark
        ]

        ys = [
            int(landmark.y * h)
            for landmark in hand_landmarks.landmark
        ]

        x1 = max(0, min(xs) - MARGIN)
        y1 = max(0, min(ys) - MARGIN)

        x2 = min(w, max(xs) + MARGIN)
        y2 = min(h, max(ys) + MARGIN)

        # ----------------------------------------------------
        # DISPLAY OVERLAY
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            "HAND DETECTED",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # ====================================================
        # SAVE CLEAN HAND IMAGE
        # ====================================================

        current_time = time.time()

        if (
            started
            and count < TARGET_IMAGES
            and current_time - last_save_time >= CAPTURE_INTERVAL
        ):

            # Crop from CLEAN frame
            # No landmarks
            # No green box
            # No text

            hand_crop = clean_frame[y1:y2, x1:x2]

            if hand_crop.size > 0:

                crop_height, crop_width = hand_crop.shape[:2]

                # ------------------------------------------------
                # MAKE SQUARE
                # ------------------------------------------------

                side = max(crop_height, crop_width)

                top = (side - crop_height) // 2
                bottom = side - crop_height - top

                left = (side - crop_width) // 2
                right = side - crop_width - left

                square_image = cv2.copyMakeBorder(
                    hand_crop,
                    top,
                    bottom,
                    left,
                    right,
                    cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )

                # ------------------------------------------------
                # RESIZE TO 224 x 224
                # ------------------------------------------------

                square_image = cv2.resize(
                    square_image,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA
                )

                # ------------------------------------------------
                # SAVE
                # ------------------------------------------------

                filename = os.path.join(
                    SAVE_FOLDER,
                    f"{LETTER}_{count + 1:03d}.jpg"
                )

                cv2.imwrite(
                    filename,
                    square_image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                count += 1
                last_save_time = current_time

                print(
                    f"Saved image {count}/{TARGET_IMAGES}"
                )

    else:

        cv2.putText(
            frame,
            "NO HAND DETECTED",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    cv2.putText(
        frame,
        f"LETTER: {LETTER}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"IMAGES: {count}/{TARGET_IMAGES}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if not started:

        cv2.putText(
            frame,
            f"Make {LETTER} and press SPACE",
            (30, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "COLLECTING...",
            (30, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "ASL Clean Image Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # SPACE = START
    if key == ord(" "):

        started = True
        print("Collection started!")

    # Q = QUIT
    if key == ord("q"):

        print("Collection stopped.")
        break

    # ========================================================
    # COLLECTION COMPLETE
    # ========================================================

    if count >= TARGET_IMAGES:

        print("--------------------------------------")
        print("COLLECTION COMPLETE!")
        print("Letter:", LETTER)
        print("Images:", count)
        print("--------------------------------------")

        cv2.waitKey(1500)
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()
hands.close()