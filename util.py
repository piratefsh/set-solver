import cv2 

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