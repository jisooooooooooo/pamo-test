import cv2
from camera import Camera
from motion_detector import MotionDetector

def main():
    camera = Camera()
    detector = MotionDetector()
    prev_motions = set()

    while True:
        success, frame = camera.read()
        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_motions = detector.detect(frame_rgb)

        new_motions = current_motions - prev_motions
        if new_motions:
            print("감지된 모션:", ", ".join(new_motions))
        prev_motions = current_motions.copy()

        frame = cv2.flip(frame, 1)
        cv2.imshow('Motion Detection', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
