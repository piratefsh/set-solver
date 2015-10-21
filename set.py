import cv2
import cv2.cv as cv
import sys
import numpy as np 
import util as util
import os

COLOR_RED = (0, 0, 255)
SIZE_CARD = (64*3, 89*3)
SIZE_CARD_W, SIZE_CARD_H = SIZE_CARD

def detect_cards(img, num_cards):
    if img is None:
        return None 

    img_binary = get_binary(img)
    contours = find_contours(img_binary)
    return transform_cards(img, contours, num_cards, draw_rects=False)

def transform_cards(img, contours, num, draw_rects=False):
    cards = []
    for i in range(num):
        card = contours[i]

        # get bounding rectangle
        rect = cv2.minAreaRect(card)
        r = cv.BoxPoints(rect)

        # convert to ints
        r = [(int(x), int(y)) for x,y in r]

        if draw_rects:
            cv2.rectangle(img, r[0], r[2], COLOR_RED)

        transformed = transform_card(card, img)
        cards.append(transformed)
    return cards

def transform_card(card, image):
    # get poly of contour
    perimeter = cv2.arcLength(card, True)
    approximated_poly_raw = cv2.approxPolyDP(card, 0.02*perimeter, True)

    # get rectified points in clockwise order
    approximated_poly = util.rectify(approximated_poly_raw)

    dest = np.array([[0,0], [SIZE_CARD_W,0], [SIZE_CARD_W,SIZE_CARD_H], [0,SIZE_CARD_H]], np.float32)
    
    # do transformatiom
    transformation = cv2.getPerspectiveTransform(approximated_poly, dest)
    warp = cv2.warpPerspective(image, transformation, SIZE_CARD)
    return warp 

def find_contours(bin_img, num=-1, return_area=False):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # if return_area:
    #     return [(x, cv2.contourArea(x)) for x in contours]

    # else:
    if num > 0: return contours[:num]
    return contours

def get_binary(img, thresh=180):
    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gaussian blur to remove noise
    img_blur = cv2.GaussianBlur(img_gray, ksize=(3,3), sigmaX=0)

    # threshold
    flag, img_threshold = cv2.threshold(img_blur, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)  

    return img_threshold

def get_card_color(card):
    blue = [pix[0] for row in card for pix in row ]
    green = [pix[1]  for row in card for pix in row ]
    red = [pix[2] for row in card for pix in row ]

    bgr = (min(blue), min(green), min(red))
    b, g, r = bgr 

    if max(bgr) == g:
        return 'green'
    if r/b > 2:
        return 'red'

    return 'purple' 

def get_card_shape(card):
    binary = get_binary(card, thresh=150)
    contours = find_contours(binary)

    contours_area = [cv2.contourArea(c) for c in contours]
    return contours

def test():
    # 3 cards on flat table
    cards_3 = cv2.imread('images/set-3-texture.jpg')
    thresh_3 = get_binary(cards_3)
    contours = find_contours(thresh_3, 3)

    assert len(transform_cards(cards_3, contours, 3)) == 3

    # 5 cards at an angle    
    cards_5_tilt = cv2.imread('images/set-5-random.jpg')
    res5 = detect_cards(cards_5_tilt, 5)
    assert res5 is not None and len(res5) == 5 

    res3 = detect_cards(cards_3, 3)
    assert res3 is not None and len(res3) == 3

    for i in range(len(res5)):
        c = res5[i]
        util.show(c, 'card')
        cv2.imwrite('images/cards/card-%d.jpg' % i, c)
    

    for link in os.listdir('images/cards'):
        img = cv2.imread('images/cards/%s' % link)
        print get_card_color(img)
        print len(get_card_shape(img))
