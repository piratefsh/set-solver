import cv2
import cv2.cv as cv
import sys
import numpy as np 
import util as util
import os
import code

COLOR_RED = (0, 0, 255)
SIZE_CARD = (64*3, 89*3)
SIZE_CARD_W, SIZE_CARD_H = SIZE_CARD

PROP_COLOR_RED = 0 
PROP_COLOR_GREEN = 1
PROP_COLOR_PURPLE = 2
PROP_COLOR_MAP = ['RED', 'GREEN', 'PURPLE']

PROP_SHAPE_DIAMOND = 0
PROP_SHAPE_OBLONG = 1
PROP_SHAPE_SQUIGGLE = 2
PROP_SHAPE_MAP = ['DIAMOND', 'OBLONG', 'SQUIGGLE']

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

def get_approx_poly(card, rectify=True):
    perimeter = cv2.arcLength(card, True)
    
    approximated_poly = cv2.approxPolyDP(card, 0.02*perimeter, True)

    # get rectified points in clockwise order
    if rectify: 
        approximated_poly = util.rectify(approximated_poly)

    return approximated_poly

def transform_card(card, image):
    # get poly of contour
    approximated_poly = get_approx_poly(card)

    dest = np.array([[0,0], [SIZE_CARD_W,0], [SIZE_CARD_W,SIZE_CARD_H], [0,SIZE_CARD_H]], np.float32)
    
    # do transformatiom
    transformation = cv2.getPerspectiveTransform(approximated_poly, dest)
    warp = cv2.warpPerspective(image, transformation, SIZE_CARD)
    return warp 

def find_contours(bin_img, num=-1, return_area=False):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if num > 0: 
        return contours[:num]
    
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

    # if mostly green
    if max(bgr) == g:
        return PROP_COLOR_GREEN

    # if a lot more red than blue, is probably red
    if r/b > 2:
        return PROP_COLOR_RED

    # else, probably purple
    return PROP_COLOR_PURPLE 

def get_card_shape(card, training_set):
    binary = get_binary(card, thresh=150)
    contours = find_contours(binary)
    poly = get_approx_poly(contours[1], rectify=False)

    # for each card in trainings set, find one with most similarity 
    diffs = []
    this_shape = get_shape_image(card)
    for i, that_shape in training_set.items():

        # resize image
        this_shape_res = util.resize(this_shape, that_shape.shape)

        # find diff and its sum
        d = cv2.absdiff(this_shape_res, that_shape)
        sum_diff = np.sum(d)

        diffs.append(sum_diff)

    # return index of shape that has minimum difference
    return diffs.index(min(diffs))

def get_shape_image(img):
    binary = get_binary(img, thresh=180)
    contours = find_contours(binary)
    shape_contour = contours[1]
    shape_img = util.draw_contour(contours, 1)
    x,y,w,h = cv2.boundingRect(shape_contour)
    cropped = shape_img[y:y+h, x:x+w]
    return cropped

def train_cards(imgs):
    training_set = {}
    # train for shapes, return contours of shapes
    for i in range(len(imgs)):
        img = imgs[i]
        shape = get_shape_only(img)
        training_set[i] = shape
    return training_set

def get_card_number(card):
    binary = get_binary(card, thresh=150)
    contours = find_contours(binary)
    poly = get_approx_poly(contours[1], rectify=False)

    # forget about first outline of card
    contours_area = [cv2.contourArea(c) for c in contours][1:]

    ratios = np.divide(contours_area, contours_area[1:] + [1])

    count = 1

    for r in ratios:
        if r > 1.1:
            break
        else:
            count += 1

    return count

def get_card_texture(card, square=20):

    binary = get_binary(card, thresh=150)
    contour = find_contours(binary)[1]

    m = cv2.moments(contour)

    cx = int(m['m10']/m['m00'])
    cy = int(m['m01']/m['m00'])

    # get bounding rectangle
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    #rect = cv2.minAreaRect(contour)
    #r = cv.BoxPoints(rect)    

    # print rect
    # r = cv.BoxPoints(rect)


    # # get reference square
    # ref_rect = cv2.getRectSubPix(card, (square,square), ((square+10)/2, (square+10)/2))
    # gray_ref_rect = cv2.cvtColor(ref_rect, cv2.COLOR_RGB2GRAY)
    

    #cv2.threshold(img_blur, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    #bin_rect = get_binary(gray_rect, thresh=200)

    cv2.rectangle(card, (x,y), (x+w,y+h), COLOR_RED)

    #r = [(int(x), int(y)) for x,y in r]

    #code.interact(local=locals())

    rect = cv2.getRectSubPix(card, (square,square), (x+w/2, y+h/2))

    print np.std(cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY))
    # gray_rect = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)

    #cv2.rectangle(card, r[0], r[2], COLOR_RED)
    util.show(rect)


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
        # util.show(c, 'card')
        cv2.imwrite('images/cards/card-5-%d.jpg' % i, c)

    for i in range(len(res3)):
        c = res3[i]
        # util.show(c, 'card')
        cv2.imwrite('images/cards/card-3-%d.jpg' % i, c)

    # train cards
    shape_diamond = cv2.imread('images/cards/card-5-4.jpg')
    shape_oblong = cv2.imread('images/cards/card-5-3.jpg')
    shape_squiggle = cv2.imread('images/cards/card-3-1.jpg')
    training_set = train_cards([shape_diamond, shape_oblong, shape_squiggle])

    # for cards detected, get properties
    for link in os.listdir('images/cards'):
        img = cv2.imread('images/cards/%s' % link)
        util.show(img)
        print PROP_COLOR_MAP[get_card_color(img)]
        print PROP_SHAPE_MAP[get_card_shape(img, training_set)]
        print get_card_number(img)
        # get_card_texture(img)
        print('---')

    print 'tests pass'
