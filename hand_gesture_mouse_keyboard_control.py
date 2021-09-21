import cv2
import mediapipe as mp
import time
import numpy as np
import autopy
import math
from time import sleep
import pyautogui
from pynput.mouse import Button, Controller

mouse = Controller()
wCam, hCam = 640, 480
frameR = 50  # Frame Reduction
smoothening = 5
wScr, hScr = autopy.screen.size()


class hand_detector:
    def __init__(
        self, mode=False, max_hands=1, detect_confidence=0.7, track_confidence=0.7
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detect_confidence = detect_confidence
        self.track_confidence = track_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode, self.max_hands, self.detect_confidence, self.track_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        # self.mouse = Controller()
        # self.wCam, self.hCam = 640, 480
        # self.frameR = 100  # Frame Reduction
        # self.smoothening = 5
        # self.wScr, self.hScr = autopy.screen.size()

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

        return img

    def find_position(self, img, hand_nos=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_nos]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                # if id % 4 == 2:
                #     cv2.circle(
                #         img, (cx, cy), 5, (0, 0, 255), cv2.FILLED
                #     )  # color is in BGR format
                # if id % 4 == 3:
                #     cv2.circle(
                #         img, (cx, cy), 5, (255, 0, 0), cv2.FILLED
                #     )  # color is in BGR format

                # if id % 4 == 0 and id != 0:  # to check id of which hand points
                #     cv2.circle(
                #         img, (cx, cy), 5, (0, 255, 0), cv2.FILLED
                #     )  # color is in BGR format

        return lm_list

    def fingers_grab(self, lm_list, hand_nos=0, draw=True):
        ans = True
        hands_up = True
        for i in range(8, 21, 4):
            if (
                lm_list[i][2] < lm_list[i - 2][2]
                and lm_list[i - 1][2] < lm_list[i - 2][2]
            ):
                ans = False

        for i in range(8, 21, 4):
            if lm_list[i][2] > lm_list[0][2]:
                hands_up = False

        return ans & hands_up

    def fingers_up(self, lm_list):
        try:
            fingers = []
            # Thumb
            print(self.tipIds[0])
            if lm_list[self.tipIds[0]][1] > lm_list[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):

                if lm_list[self.tipIds[id]][2] < lm_list[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # totalFingers = fingers.count(1)
        except Exception as e:
            print(e)
        return fingers

    def finger_distance(self, p1, p2, img, lm_list, draw=True, r=15, t=3):
        x1, y1 = lm_list[p1][1:]
        x2, y2 = lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():

    ptime = 0
    ctime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    wScr, hScr = autopy.screen.size()

    detector = hand_detector()
    not_grabbed = True
    tc = 20
    while tc:
        # tc -= 1
        try:
            success, img = cap.read()

            img = detector.find_hands(img)
            lm_list = detector.find_position(img)
            fingers = detector.fingers_up(lm_list)

            cv2.rectangle(
                img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 255, 0), 2
            )

            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]
                x2, y2 = lm_list[12][1:]

            if fingers[1] & fingers[2] & fingers[3] & fingers[4]:
                mouse.release(Button.left)
                not_grabbed = True

            if fingers[1] and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                mouse.release(Button.left)
                not_grabbed = False

            # 4. Only Index Finger : Moving Mode
            if (fingers[1] == 1 and fingers[2] == 0) or detector.fingers_grab(lm_list):
                # 5. Convert Coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                # 6. Smoothen Values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # 7. Move Mouse
                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 8. Both Index and middle fingers are up : Clicking Mode
            if (
                fingers[1] == 1
                and fingers[2] == 1
                and fingers[3] == 0
                and fingers[4] == 0
            ):
                # 9. Find distance between fingers
                length, img, lineInfo = detector.finger_distance(8, 12, img, lm_list)
                print(length)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(
                        img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED
                    )
                    autopy.mouse.click()

            if (
                fingers[0] == 0
                and fingers[2] == 0
                and fingers[1] == 1
                and fingers[3] == 1
                and fingers[4] == 1
            ):
                sleep(0.2)
                pyautogui.keyDown("altleft")
                pyautogui.keyDown("f4")
                pyautogui.keyUp("altleft")
                pyautogui.keyUp("f4")
                pyautogui.keyDown("esc")
                pyautogui.keyUp("esc")

            if detector.fingers_grab(lm_list):
                if not_grabbed:
                    mouse.press(Button.left)
                    not_grabbed = False

                cv2.putText(
                    img,
                    "Object grabbed...",
                    (60, 40),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    3,
                )

            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime

            cv2.putText(
                img,
                str(int(fps)),
                (10, 40),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 255, 255),
                3,
            )

            cv2.imshow("Image", img)
            cv2.waitKey(1)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
