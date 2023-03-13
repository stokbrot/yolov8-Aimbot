import serial
import os
from ultralytics import YOLO
import cv2
from ultralytics.yolo.utils.plotting import Annotator
import mss
import time
from roboflow import Roboflow
import numpy as np
import serial
import mouse
import win32api
import keyboard
import win32api
import win32con
import win32gui
import win32ui

conf = 0.7
speed = 2
AIMING_POINT = 3  # 0 for "head", 1 for chest, 2 for legs, anything else for middle
ScreenSizeX = 1920
ScreenSizeY = 1080
Fov = 300

# model = YOLO("yolov8n.pt")
model = YOLO("best.pt")


def clear(): return os.system('cls')


# open port of arduino
arduino = serial.Serial('COM6', 115200, timeout=0)

closest = []
fps_list = []

# screen grab copied from somewhere /// 1 fps faster than mss


def grab_screen(region=None):
    hwin = win32gui.GetDesktopWindow()

    if region:
        left, top, x2, y2 = region
        widthScr = x2 - left + 1
        heightScr = y2 - top + 1
    else:
        widthScr = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        heightScr = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, widthScr, heightScr)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (widthScr, heightScr),
                 srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (heightScr, widthScr, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


with mss.mss() as sct:
    Wd, Hd = sct.monitors[1]["width"], sct.monitors[1]["height"]

    monitor = (int(Wd / 2 - Fov / 2),
               int(Hd / 2 - Fov / 2),
               int(Wd / 2 + Fov / 2),
               int(Hd / 2 + Fov / 2))


while True:
    t = time.time()

    # reset closes list
    closest.clear()

    img = np.array(grab_screen(region=monitor))

    results = model.predict(img, verbose=False)

    for r in results:

        annotator = Annotator(img)

        boxes = r.boxes
        for id_box, box in enumerate(boxes):
            if box.conf > conf:
                global target
                # get box coordinates in (top, left, bottom, right) format
                rl = box.xyxy[0]
                c = box.cls
                annotator.box_label(rl, model.names[int(c)])

                x1 = float(rl[0])
                x2 = float(rl[2])
                y1 = float(rl[1])
                y2 = float(rl[3])

                height = int(y2 - y1)

                if AIMING_POINT == 0:
                    height = (height / 8) * 1

                elif AIMING_POINT == 2:
                    height = (height / 8) * 5

                elif AIMING_POINT == 1:
                    height = (height / 8) * 2
                else:
                    height = height-(height/2)

                Xenemycoord = (x2 - x1) / 2 + x1
                Yenemycoord = y1+height

                difx = int(Xenemycoord - (Fov / 2)) * speed
                dify = int(Yenemycoord - (Fov / 2)) * speed

                # If on target:=
                # if (dify < (height*0.8) > -dify) or (difx < ((x1-x2)*0.8) > -difx):

                if abs(difx) < 2 and abs(dify) < 2:
                    pass

                distance = ((difx**2)+(difx**2))**0.5

                if all(di > distance for di in closest):
                    min_difx = difx
                    min_dify = dify

                closest.append(distance)

    if keyboard.is_pressed('v'):

        # Send data to Arduino (use ur driver here idk)
        data = str(min_difx) + ':' + str(min_dify)
        arduino.write(data.encode())

    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)

    cv2.waitKey(1)
    if keyboard.is_pressed('p'):
        cv2.destroyAllWindows()

    fps_list.append(int(1 / (time.time() - t)))

    # avrg fps
    if len(fps_list) > (int(1 / (time.time() - t)))*2:

        clear()
        print('fps:', int(sum(fps_list)/len(fps_list)))
        fps_list.clear()
