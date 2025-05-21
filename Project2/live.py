import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils import threshold, ball_separator
from typing import List, Any
from cv2.typing import MatLike
from numpy.typing import NDArray

WHITE: tuple[int,int,int] = (142, 142, 142)
BLUE: tuple[int,int,int] = (253, 128, 0)
GREEN: tuple[int,int,int] = (0, 255, 0)
PINK: tuple[int, int, int] = (180, 175, 236)
BROWN: tuple[int, int, int] = (0, 78, 149)
YELLOW: tuple[int, int, int] = (0, 241, 253)
BLACK: tuple[int, int, int] = (0, 0, 0)
RED: tuple[int, int, int] = (0, 0, 255)


def show_direction(frame, ball_start, ball_pos, color):
    line = ball_pos - ball_start
    cv2.line(frame, ball_pos, ball_pos + line, color, 2)
    if np.linalg.norm(line) > 2:
        return True
    return False

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
def track_ball(coordinates, start_coordinate, frame, color, show=False) -> tuple[List[int], bool]:
    state: bool = False
    if coordinates.shape[0] != 0:
        if show:
            print(coordinates[np.argmin(np.linalg.norm(coordinates - start_coordinate, axis=1))])
        new_coordinate = coordinates[np.argmin(np.linalg.norm(coordinates - start_coordinate, axis=1))]
        cv2.circle(frame, (new_coordinate[0], new_coordinate[1]), 15, color, 1)
        cv2.circle(frame, (new_coordinate[0], new_coordinate[1]), 1, color, 2)
        state = show_direction(frame, start_coordinate, new_coordinate, color)
        start_coordinate = new_coordinate
    return start_coordinate, state

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

    green = blue = pink = black = brown = yellow = True
    first_frame = True
    move = False
    steady = 0
    color_ball = {}
    score = 0

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
            white_coor: NDArray[Any] = show_balls(frame, white_ball, WHITE, WHITE, show=False)

            if blue:
                blue_coor: NDArray[Any] = show_balls(frame, blue_ball, BLUE, BLUE, low=0, show=False)
                color_ball["blue"] = blue_coor
                track_blue, _ = track_ball(blue_coor, track_blue, frame, BLUE)
            if green:
                green_coor: NDArray[Any] = show_balls(frame, green_ball, GREEN, GREEN, low=70, show=False)
                color_ball["green"] = green_coor
                track_green, _ = track_ball(green_coor, track_green, frame, GREEN)
            if pink:
                pink_coor: NDArray[Any] = show_balls(frame, pink_ball, PINK, PINK, show=False)
                color_ball["pink"] = pink_coor
                track_pink, _ = track_ball(pink_coor, track_pink, frame, PINK)
            if brown:
                brown_coor: NDArray[Any] = show_balls(frame, brown_ball, BROWN, BROWN, low=150, show=False)
                color_ball["brown"] = brown_coor
                track_brown, _ = track_ball(brown_coor, track_brown, frame, BROWN)
            if yellow:
                yellow_coor: NDArray[Any] = show_balls(frame, yellow_ball, YELLOW, YELLOW, show=False)
                color_ball["yellow"] = yellow_coor
                track_yellow, _ = track_ball(yellow_coor, track_yellow, frame, YELLOW)
            if black:
                black_coor: NDArray[Any] = show_balls(frame, black_ball, BLACK, BLACK, show=False)
                color_ball["black"] = black_coor
                track_black, _ = track_ball(black_coor, track_black, frame, BLACK)
            red_coor: NDArray[Any] = show_balls(frame, red_balls, RED, RED)

            if first_frame:
                len_red = len(red_coor)+1
                first_frame = False

            track_white, state = track_ball(white_coor, track_white, frame, (142, 142, 142))

            if state:
                if move:
                    move = not move
                    print("Ruch trwa")
            else:
                steady += 1
                if steady >= 25:
                    if not move:
                        move = not move
                        if len_red > len(red_coor): # czasami działa częściej nie
                            print("Wbito czerwoną")
                            score += 1
                        print("Koniec ruchu")
                        print(f"Aktualny wynik: {score}")
                        len_red = len(red_coor)
                        len_color = len(color_ball)
                    steady = 0

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