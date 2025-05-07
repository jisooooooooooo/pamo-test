import mediapipe as mp
import math


class MotionDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(min_detection_confidence=0.7)

        self.x_history = []
        self.nose_x_history = []
        self.direction_history = []

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def detect(self, frame_rgb):
        current_motions = set()

        # 손 모션 감지
        hand_results = self.hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # 손을 위로 들기
                if index_tip.y < wrist.y - 0.15:
                    current_motions.add("손을 위로 들기")

                # 손을 좌우로 흔들기
                self.x_history.append(index_tip.x)
                if len(self.x_history) > 10:
                    self.x_history.pop(0)
                if len(self.x_history) == 10:
                    if max(self.x_history) - min(self.x_history) > 0.3:
                        current_motions.add("손을 좌우로 흔들기")
        else:
            self.x_history = []

        # 고개 좌우 흔들기 감지
        face_results = self.face.process(frame_rgb)
        if face_results.multi_face_landmarks:
            nose = face_results.multi_face_landmarks[0].landmark[1]
            self.nose_x_history.append(nose.x)

            if len(self.nose_x_history) > 15:
                self.nose_x_history.pop(0)

            if len(self.nose_x_history) >= 2:
                delta = self.nose_x_history[-1] - self.nose_x_history[-2]
                if delta > 0.005:
                    self.direction_history.append("right")
                elif delta < -0.005:
                    self.direction_history.append("left")
                else:
                    self.direction_history.append("none")

                if len(self.direction_history) > 10:
                    self.direction_history.pop(0)

                if (
                        len(self.direction_history) >= 3 and
                        (
                                (self.direction_history[-3] == "left" and
                                 self.direction_history[-2] == "right" and
                                 self.direction_history[-1] == "left") or
                                (self.direction_history[-3] == "right" and
                                 self.direction_history[-2] == "left" and
                                 self.direction_history[-1] == "right")
                        )
                ):
                    current_motions.add("고개 좌우로 흔들기")
                    self.direction_history.clear()

        return current_motions
