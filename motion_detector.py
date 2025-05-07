import mediapipe as mp
import math

class MotionDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

    def detect(self, frame_rgb):
        current_motions = set()
        hand_results = self.hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                if index_tip.y < wrist.y - 0.15:
                    current_motions.add("손을 위로 들기")
        return current_motions
