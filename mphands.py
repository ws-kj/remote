import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import math
import os



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

xmin, xmax, ymin, ymax = 0, 0, 0, 0
curcal = 0

dbase = 0
dmult = 1
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width = image.shape[1]
        height = image.shape[0]
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                subset = landmark_pb2.NormalizedLandmarkList(
                    landmark = [
                        hand.landmark[0],   #wrist
                        hand.landmark[5],   #index knuckle
                  #      hand.landmark[6],
                  #      hand.landmark[7],
                        hand.landmark[8],   #index tip
                        hand.landmark[9],   #middle knuckle
                    ]
                )

                ix = hand.landmark[8].x * width
                iy = hand.landmark[8].y * height
                wx = hand.landmark[5].x * width
                wy = hand.landmark[5].y * height
                mx = hand.landmark[9].x * width
                my = hand.landmark[9].y * width

                sx = float(ix-wx)
                sy = float(iy-wy)
                
                fx, fy = ix, iy
                while fx+sx > 0 and fx+sx < width and fy+sy > 0 and fy+sy < height: 
                    fx+=sx
                    fy+=sy

                mp_drawing.draw_landmarks(image, subset, None,
                    mp_drawing.DrawingSpec(color=(217, 133, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(105, 0, 101), thickness=2, circle_radius=2),)

                line = cv2.line(image, (int(wx), int(wy)), (int(fx), int(fy)), (0, 255, 0), 2)
                
                if cv2.waitKey(10) & 0xFF == ord('w') and curcal < 2:
                    if curcal == 0:
                        xmin, ymax = ix, iy
                    if curcal == 1:
                        xmax, ymin = ix, iy
                    curcal += 1
                    dbase = int((math.sqrt((wx-mx)*(wx-mx)-(wy-my))))

                if curcal > 1:
                    image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
          
                    dmult = math.sqrt((wx-mx)*(wx-mx)-(wy-my))/dbase
                    dbase = dmult*dbase

                    w = (xmax-xmin)*dmult 
                    h = (ymax-ymin)*dmult
                    xmin, xmax = wx-(w/2), wx+(w/2)
                    ymin, ymax = wy-(h/2), wy+(h/2)

                    realx = (ix-xmin)/(xmax-xmin)*1366
                    realy = (iy-ymax)/(ymin-ymax)*768
                    print(realx)
                    print(realy)
                    print("\n")
                    com = "xdotool mousemove " + str(realx) + " " + str(realy)
                    os.system(com)

                            
        cv2.imshow("Image", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
mp_drawing.DrawingSpec()

