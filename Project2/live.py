import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import threshold, ball_separator
from typing import List, Any
from cv2.typing import MatLike
from numpy.typing import NDArray

def show_balls(frame, balls, inner_color, outher_color, low=50, high=400, show=True) -> NDArray[Any]:
    contours, _ = cv2.findContours(balls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours: List[Any] = [c for c in contours if low < cv2.contourArea(c) < high]
    balls_coordinates: List[Any] = []
    if contours:
        for c in filtered_contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if cx < 125 or cx > 1175 or cy < 95 or cy > 615:
                    continue
                if show:
                    cv2.circle(frame, (cx, cy), 15, outher_color, 1)
                    cv2.circle(frame, (cx, cy), 1, inner_color, 2)
                balls_coordinates.append([cx, cy])
        return np.array(balls_coordinates)
    return np.array([])
def track_ball(coordinates, start_coordinate, frame, color, show=False) -> List[int]:
    if coordinates.shape[0] != 0:
        if show:
            print(coordinates[np.argmin(np.linalg.norm(coordinates - start_coordinate, axis=1))])
        start_coordinate = coordinates[np.argmin(np.linalg.norm(coordinates - start_coordinate, axis=1))]
        cv2.circle(frame, (start_coordinate[0], start_coordinate[1]), 15, color, 1)
        cv2.circle(frame, (start_coordinate[0], start_coordinate[1]), 1, color, 2)
    return start_coordinate

def main() -> None:
    # Tworzenie obiektu do przechwytywania wideo
    video = cv2.VideoCapture('video.mp4')

    # Inicjacja zapisu współrzędnych bil
    track_white: List[int] = [0, 0]
    track_blue: List[int] = [0, 0]
    track_green: List[int] = [0, 0]
    track_pink: List[int] = [0, 0]
    track_brown: List[int] = [0, 0]
    track_yellow: List[int] = [0, 0]
    track_black: List[int] = [0, 0]

    while(video.isOpened()):
        # Czytanie klatki
        ret, frame = video.read()

        if ret == True:
            # Ekstrakcja bil z obrazu
            white_ball: MatLike = ball_separator(frame,"white")
            red_balls: MatLike = ball_separator(frame, "red")
            yellow_ball: MatLike = ball_separator(frame, "yellow")
            pink_ball: MatLike = ball_separator(frame,"pink")
            green_ball: MatLike = ball_separator(frame,"green")
            black_ball: MatLike = ball_separator(frame,"black")
            blue_ball: MatLike = ball_separator(frame,"blue")
            brown_ball: MatLike = ball_separator(frame, "brown")

            # Wyciąganie współrzędnych bil
            white_coor: NDArray[Any] = show_balls(frame, white_ball, (142, 142, 142), (142, 142, 142), show=False)
            blue_coor: NDArray[Any] = show_balls(frame, blue_ball, (253, 128, 0), (253, 128, 0), low=0, show=False)
            green_coor: NDArray[Any] = show_balls(frame, green_ball, (0, 255, 0), (0, 255, 0), low=70, show=False)
            pink_coor: NDArray[Any] = show_balls(frame, pink_ball, (180, 175, 236), (180, 175, 236), show=False)
            brown_coor: NDArray[Any] = show_balls(frame, brown_ball, (0, 78, 149), (0, 78, 149), low=150, show=False)
            yellow_coor: NDArray[Any] = show_balls(frame, yellow_ball, (0, 241, 253), (0, 241, 253), show=False)
            black_coor: NDArray[Any] = show_balls(frame, black_ball, (0, 0, 0), (0, 0, 0), show=False)
            red_coor: NDArray[Any] = show_balls(frame, red_balls, (0, 0, 255), (0, 0, 255))

            # Śledzenie bil
            track_white = track_ball(white_coor, track_white, frame, (142, 142, 142), show=True)
            track_blue = track_ball(blue_coor, track_blue, frame, (253, 128, 0))
            track_green = track_ball(green_coor, track_green, frame, (0, 255, 0))
            track_pink = track_ball(pink_coor, track_pink, frame, (180, 175, 236))
            track_brown = track_ball(brown_coor, track_brown, frame, (0, 78, 149))
            track_yellow = track_ball(yellow_coor, track_yellow, frame, (0, 241, 253))
            track_black = track_ball(black_coor, track_black, frame, (0, 0, 0))

            # Pokazywanie zedytowanej klatki
            cv2.imshow('Frame', frame)

            # Zamknięcie programu po kliknięciu "q"
            if  cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()