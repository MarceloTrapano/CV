import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import threshold, ball_separator
def show_balls(frame, balls, inner_color, outher_color, low=50, high=400):
    contours, _ = cv2.findContours(balls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [c for c in contours if low < cv2.contourArea(c) < high]
    if contours:
# Zakładamy, że największy kontur to kulka
        for c in filtered_contours:
        # Oblicz momenty
            M = cv2.moments(c)

            if M["m00"] != 0:
                # Centrum kulki:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if cx < 125 or cx > 1175 or cy < 95 or cy > 615:
                    continue
                # Opcjonalnie: narysuj punkt
                cv2.circle(frame, (cx, cy), 20, outher_color, 1)
                cv2.circle(frame, (cx, cy), 1, inner_color, 2)
def main():
    video = cv2.VideoCapture('video.mp4')
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            white_ball = ball_separator(frame,"white")
            red_balls = ball_separator(frame, "red")
            yellow_ball = ball_separator(frame, "yellow")
            pink_ball = ball_separator(frame,"pink")
            green_ball = ball_separator(frame,"green")
            black_ball = ball_separator(frame,"black")
            blue_ball = ball_separator(frame,"blue")
            brown_ball = ball_separator(frame, "brown")
            show_balls(frame, white_ball, (142, 142, 142), (142, 142, 142))
            show_balls(frame, blue_ball, (253, 128, 0), (253, 128, 0), low=0)
            show_balls(frame, green_ball, (0, 255, 0), (0, 255, 0), low=70)
            show_balls(frame, pink_ball, (180, 175, 236), (180, 175, 236))
            show_balls(frame, brown_ball, (0, 78, 149), (0, 78, 149), low=150)
            show_balls(frame, yellow_ball, (0, 241, 253), (0, 241, 253))
            show_balls(frame, black_ball, (0, 0, 0), (0, 0, 0))
            show_balls(frame, red_balls, (0, 0, 255), (0, 0, 255))
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()