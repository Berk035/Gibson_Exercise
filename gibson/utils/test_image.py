import cv2 as cv
import os.path as osp
import os
import sys
import numpy as np


def add_text(img):
    font = cv.FONT_HERSHEY_SIMPLEX
    # font = cv2.FONT_HERSHEY_PLAIN
    # x,y,z = self.robot.get_position()
    # r,p,ya = self.robot.get_rpy()

    cv.putText(img, 'ObsDist:{0:.3f}'.format(np.mean(img[0:128,0:128])), (10,120), font, 0.3, (255, 255, 255), 1, cv.LINE_AA)
    # cv2.putText(img, 'x:{0:.2f} y:{1:.2f} z:{2:.2f}'.format(x,y,z), (10, 100), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, 'ro:{0:.4f} pth:{1:.4f} ya:{2:.4f}'.format(r,p,ya), (10, 40), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, 'potential:{0:.4f}'.format(self.potential), (10, 60), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, 'fps:{0:.4f}'.format(self.fps), (10, 80), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    path_1="/home/berk/PycharmProjects/Gibson_Exercise/gibson/utils/depth_images"
    i=1
    while(True):
        img = cv.imread(os.path.join(path_1, 'Frame_%d.jpg')%i)
        sz = 128
        screen_half = int(sz / 2) #64
        height_offset = int(sz / 4) #32
        screen_delta =  int(sz / 8) #16
        clip = img[screen_half + height_offset - screen_delta: screen_half + height_offset + screen_delta,
               screen_half - screen_delta: screen_half + screen_delta, -1]
        clip2 = img[0:128,0:128]
        img = add_text(img)
        if img is None:
            sys.exit("Could not read the image.")
        cv.imshow("Display window", img)
        k = cv.waitKey(0)
        if k == ord("q"):
            sys.exit("Terminated")
        elif k== ord("n"):
            i+=1
            continue