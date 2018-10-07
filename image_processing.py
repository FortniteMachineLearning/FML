import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import math
import os
from random import randint
from Annotate_Images_To_Xml import write_xml


options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.3,
    'gpu': 1.0
}

# GLOBAL VAIABLES
PERSON = 'person'
tl_list = []
br_list = []
obj_list = []
FORTNITE_CHARACTER = 'fortnite_character'
path = 'images\Fortnite_Characters'

tfnet = TFNet(options)

capture = cv2.VideoCapture('Fortnite_Pregame_Lobby_Reduced.avi')
colors = ([(0, 0, 255) for i in range(5)])

width = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)   # float
height = capture.get(cv2.CAP_PROP_FRAME_WIDTH)  # float

y_threshold = math.floor(height / 2.5)
confidence_threshold = 0.40

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    frame_with_graphics = frame.copy()
    results = tfnet.return_predict(frame)

    if ret:
        for color, result in zip(colors, results):
            if(result['label'] != PERSON):
                break
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            if(result['bottomright']['y'] < y_threshold):
                if(result['confidence'] > confidence_threshold):
                    frame_with_graphics = cv2.rectangle(frame, tl, br, color, 1)
                    tl_list.append((int(result['topleft']['x']), int(result['topleft']['y'])))
                    br_list.append((int(result['bottomright']['x']), int(result['bottomright']['y'])))
                    obj_list.append(FORTNITE_CHARACTER)
                label = result['label']
        # if(len(obj_list) != 0):
        #     random_number = randint(0, 100000000)  # randint is inclusive at both ends
        #     img_name = "img_%d.png" % random_number
        #     cv2.imwrite(os.path.join(path, img_name), frame)
        #     img = [im for im in os.scandir(path) if str(random_number) in im.name][0]
        #     write_xml(path, img, obj_list, tl_list, br_list, savedir='annotations')
        #     obj_list = []
        #     tl_list = []
        #     br_list = []
        #     image_name = None
        # frame = cv2.rectangle(frame, tl, br, color, 1)
        # frame = cv2.putText(frame, 'Kill Him!', tl, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('frame', frame_with_graphics)
        print('FPS {:1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break
