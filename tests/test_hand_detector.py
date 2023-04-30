import os
import cv2
from hand_detector.hand_detector import HandDetector


dir = "./imgs/hands"

hand_detector = HandDetector("./hand-detector-model")


for item in os.listdir(dir):
    img = cv2.cvtColor(cv2.imread(os.path.join(dir, item)), cv2.COLOR_BGR2RGB)
    output = hand_detector.predict(img)
    print(item, output)
