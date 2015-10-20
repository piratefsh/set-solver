import cv2
import cv2.cv as cv
import sys
import numpy as np 
import util as util

def detect_cards(img):

    return []

def find_contours(bin_img, num):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num]

    return contours

def get_binary(img, thresh=180):
    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian blur to remove noise
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=0)

    # threshold
    flag, img_threshold = cv2.threshold(img_blur, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)  

    return img_threshold

def test():
    cards_3 = cv2.imread('images/set-3-texture.jpg')
    thresh_3, window = get_binary(cards_3), 'thesholded image'
    contours = find_contours(thresh_3, -1)

    cv2.drawContours(cards_3, contours, -1, (100,100,100), thickness=2)
    print(util)
    util.show(cards_3, window)
    
    res = detect_cards(cards_3) 
    assert len(res) == 3
    
    cards_5 = cv2.imread('images/set-5-random.jpg')
    res_cards_5 = detect_cards(cards_5) 
    assert len(res_cards_5) == 5