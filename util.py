import cv2 
import numpy as np

def show(img, window_name):
    # destroy existing window
    destroy(window_name)

    # show it
    cv2.imshow(window_name, img)

    # wait for key, then destroy it
    cv2.waitKey(0)
    destroy(window_name)

    return window_name

def destroy(window_name):
    cv2.destroyWindow(window_name)

# (Stolen) utility code from 
# http://git.io/vGi60A
def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew
