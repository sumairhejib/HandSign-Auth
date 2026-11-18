"""
app.py  -  Live hand-gesture human verification (local webcam app)
Sumair Hejib | 20241CAI0092 | Neural Networks And Fuzzy Logic (CAI 2512)

Prerequisite:  python train_model.py   (creates gesture_model.keras)
Run:           python app.py
Keys:          R = new challenge, Q or ESC = quit

How it works
  1. A random challenge of 4 gestures (finger counts 0-5) is shown.
  2. MediaPipe finds 21 hand landmarks in every webcam frame.
  3. The trained neural network classifies the gesture; a rule-based module counts raised fingers.
  4. Hold the requested gesture steadily for a moment to pass the step. Each step has a time limit.
  5. All 4 steps passed = VERIFIED, otherwise FAILED.

Testing without a camera:  python app.py --video some_video.mp4 [--headless --save-prefix out]
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import argparse, random, sys
import numpy as np
import cv2
import mediapipe as mp
from tensorflow import keras

CLASSES = [0, 1, 2, 3, 4, 5]
MIN_CONF = 0.70          # minimum network confidence for a matching frame
GREEN, RED, YELLOW, WHITE, GREY = (80, 200, 80), (60, 60, 230), (0, 200, 255), (255, 255, 255), (120, 120, 120)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ---------- features (same maths as train_model.py) ----------
def normalize(lm):
    p = lm - lm[0]
    s = np.max(np.linalg.norm(p[:, :2], axis=1)) + 1e-6
    return (p / s).reshape(-1).astype(np.float32)


def landmarks_to_array(hand, w, h):
    """21x3 array. x and y are scaled by the frame size so a non-square webcam frame is not distorted
    (the training photos were square)."""
    s = float(max(w, h))
    return np.array([[p.x * w / s, p.y * h / s, p.z] for p in hand.landmark], dtype=np.float32)


def finger_states(lm, thumb_margin=1.05):
    xy = lm[:, :2]
    wrist = xy[0]
    st = [int(np.linalg.norm(xy[4] - xy[17]) > thumb_margin * np.linalg.norm(xy[3] - xy[17]))]
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        st.append(int(np.linalg.norm(xy[tip] - wrist) > np.linalg.norm(xy[pip] - wrist)))
    return st            # [thumb, index, middle, ring, pinky]


# ---------- verification session ----------
def new_challenge(n=4):
    seq = [random.choice(CLASSES)]
    while len(seq) < n:
        c = random.choice(CLASSES)
        if c != seq[-1]:
            seq.append(c)
    return seq


class Session:
    def __init__(self, challenge, hold_frames, timeout_frames):
        self.challenge, self.hold_needed, self.timeout = challenge, hold_frames, timeout_frames
        self.step, self.hold, self.frames = 0, 0, 0
        self.results = []            # True/False for each finished step
        self.done, self.verified = False, False

    def update(self, pred, conf):
        if self.done:
            return
        self.frames += 1
        if pred == self.challenge[self.step] and conf >= MIN_CONF:
            self.hold += 1
        else:
            self.hold = 0
        if self.hold >= self.hold_needed:
            self.results.append(True)
            self.step += 1
            self.hold, self.frames = 0, 0
            if self.step == len(self.challenge):
                self.done, self.verified = True, True
        elif self.frames >= self.timeout:
            self.results.append(False)
            self.done, self.verified = True, False

    def abort(self):
        if not self.done:
            self.results.append(False)
            self.done, self.verified = True, False


# ---------- drawing ----------
def put(img, text, org, scale=0.8, color=WHITE, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_ui(frame, sess, hand, pred, conf, fingers):
    img = frame.copy()
    if hand is not None:
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
    H, W = img.shape[:2]
    # top bar: challenge
    cv2.rectangle(img, (0, 0), (W, 70), (40, 30, 20), -1)
    if not sess.done:
        n = sess.challenge[sess.step]
        put(img, f"Step {sess.step + 1}/{len(sess.challenge)}: show {n} finger{'s' if n != 1 else ''}", (15, 33), 0.9, YELLOW)
    else:
        put(img, "Challenge finished", (15, 33), 0.9, YELLOW)
    # progress squares
    for i, c in enumerate(sess.challenge):
        x = 15 + i * 62
        col = GREY
        if i < len(sess.results):
            col = GREEN if sess.results[i] else RED
        elif i == sess.step and not sess.done:
            col = YELLOW
        cv2.rectangle(img, (x, 42), (x + 54, 64), col, -1)
        put(img, str(c), (x + 19, 61), 0.55, (0, 0, 0), 1)
    # bottom bar: live detection
    cv2.rectangle(img, (0, H - 70), (W, H), (40, 30, 20), -1)
    if pred is None:
        put(img, "No hand detected", (15, H - 40), 0.8, RED)
    else:
        put(img, f"Gesture: {pred} ({conf * 100:.0f}%)   Fingers raised: {fingers}", (15, H - 40), 0.8, GREEN)
    if not sess.done:
        frac = min(1.0, sess.hold / sess.hold_needed)
        cv2.rectangle(img, (15, H - 28), (15 + 240, H - 12), GREY, 1)
        cv2.rectangle(img, (15, H - 28), (15 + int(240 * frac), H - 12), GREEN, -1)
        put(img, "hold steady", (265, H - 14), 0.5, WHITE, 1)
        left = max(0, sess.timeout - sess.frames)
        put(img, f"time left: {left}", (W - 170, H - 14), 0.5, WHITE, 1)
    else:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, H // 2 - 75), (W, H // 2 + 75), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
        msg, col = ("VERIFIED - HUMAN", GREEN) if sess.verified else ("VERIFICATION FAILED", RED)
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        put(img, msg, ((W - tw) // 2, H // 2 + 10), 1.3, col, 3)
        put(img, "R = new challenge    Q = quit", (W // 2 - 150, H // 2 + 50), 0.65, WHITE, 1)
    return img


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Live hand-gesture human verification")
    ap.add_argument("--model", default="gesture_model.keras")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="video file instead of a webcam (for testing)")
    ap.add_argument("--hold", type=int, default=12, help="frames the gesture must be held")
    ap.add_argument("--timeout", type=int, default=240, help="frames allowed per step (about 8 s at 30 fps)")
    ap.add_argument("--seed", type=int, help="fix the random challenge")
    ap.add_argument("--headless", action="store_true", help="no window (testing)")
    ap.add_argument("--save-prefix", help="save screenshots as PREFIX_progress.png and PREFIX_result.png")
    a = ap.parse_args()

    if not os.path.exists(a.model):
        sys.exit(f"Model file '{a.model}' not found. Run:  python train_model.py")
    model = keras.models.load_model(a.model)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        sys.exit("Could not open the webcam/video. Check the camera index (--camera) or close other apps using it.")
    live = a.video is None

    if a.seed is not None:
        random.seed(a.seed)
    sess = Session(new_challenge(), a.hold, a.timeout)
    print("Challenge:", sess.challenge)
    saved_progress = False
    while True:
        ok, frame = cap.read()
        if not ok:
            sess.abort()
            if not a.headless and not live:
                break
            if a.headless:
                break
            print("Camera frame not received.")
            break
        if live:
            frame = cv2.flip(frame, 1)          # mirror view
        if frame.shape[1] < 640:                # enlarge small frames for readability
            k = 640 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=k, fy=k)
        h, w = frame.shape[:2]
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand, pred, conf, fingers = None, None, 0.0, None
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = landmarks_to_array(hand, w, h)
            p = model.predict(normalize(lm)[None], verbose=0)[0]
            pred, conf, fingers = int(p.argmax()), float(p.max()), int(sum(finger_states(lm)))
        sess.update(pred, conf)
        out = draw_ui(frame, sess, hand, pred, conf, fingers)

        if a.save_prefix and not saved_progress and sess.step == 1 and sess.hold >= 2:
            cv2.imwrite(f"{a.save_prefix}_progress.png", out); saved_progress = True
        if sess.done:
            if a.save_prefix:
                cv2.imwrite(f"{a.save_prefix}_result.png", out)
            print("RESULT:", "VERIFIED" if sess.verified else "FAILED", "| steps:", sess.results)
            if a.headless or not live:
                if not a.headless:
                    cv2.imshow("Gesture verification", out); cv2.waitKey(2000)
                break
            # interactive: wait for R / Q
            cv2.imshow("Gesture verification", out)
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):
                    cap.release(); cv2.destroyAllWindows(); return
                if key == ord("r"):
                    sess = Session(new_challenge(), a.hold, a.timeout)
                    print("New challenge:", sess.challenge)
                    break
            continue
        if not a.headless:
            cv2.imshow("Gesture verification", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                sess = Session(new_challenge(), a.hold, a.timeout)
                print("New challenge:", sess.challenge)
    cap.release()
    if not a.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
