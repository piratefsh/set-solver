import cv2 
import numpy as np

def show(img, window_name='main'):
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
    try:
        h = h.reshape((4,2))
    except ValueError:
        return np.array([None])
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

# draw contour on empty image
def draw_contour(c, i):
    dest = np.zeros((500,300), np.float32)
    cv2.drawContours(dest, c, i, 255, cv2.cv.CV_FILLED)
    return dest

def resize(src, shape):
    dest = cv2.resize(src, (shape[1], shape[0]))
    return dest
