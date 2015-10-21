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

PROP_TEXTURE_STRIPED = 0
PROP_TEXTURE_EMPTY = 1
PROP_TEXTURE_SOLID = 2
PROP_TEXTURE_MAP = ['STRIPED', 'EMPTY', 'SOLID']

def get_card_properties(cards, training_set):
    properties = []
    for img in cards:
        num =  get_card_number(img)
        color = get_card_color(img)
        shape = get_card_shape(img, training_set)
        texture = get_card_texture(img)
        p = (num, color, shape, texture)
        properties.append(p)
    return properties

def pretty_print_properties(properties):
    for p in properties:
        num, color, shape, texture = p 
        print '%d %s %s %s' % (num, PROP_COLOR_MAP[color],\
         PROP_SHAPE_MAP[shape], PROP_TEXTURE_MAP[texture])



def detect_cards(img, draw_rects = False):
    if img is None:
        return None 

    img_binary = get_binary(img)
    contours = find_contours(img_binary)
    num_cards = get_dropoff([cv2.contourArea(c) for c in contours], maxratio=1.5)

    return transform_cards(img, contours, num_cards, draw_rects=draw_rects)

def transform_cards(img, contours, num, draw_rects=False):
    cards = []
    for i in xrange(num):
        card = contours[i]

        # get bounding rectangle
        rect = cv2.minAreaRect(card)
        r = cv.BoxPoints(rect)

        # convert to ints
        r = [(int(x), int(y)) for x,y in r]

        if draw_rects:
            cv2.rectangle(img, r[0], r[2], COLOR_RED)

        try:
            transformed = transform_card(card, img)
            cards.append(transformed)
        except:
            print 'Error processing card!! :o'
            pass

    return cards

def transform_card(card, image):
    # find out if card is rotated
    x, y, w, h = cv2.boundingRect(card) 
    card_shape = [[0,0], [SIZE_CARD_W,0], [SIZE_CARD_W, SIZE_CARD_H], [0,SIZE_CARD_H]]

    # get poly of contour
    approximated_poly = get_approx_poly(card)
    dest = np.array(card_shape, np.float32)
    
    # do transformatiom
    transformation = cv2.getPerspectiveTransform(approximated_poly, dest)
    warp = cv2.warpPerspective(image, transformation, SIZE_CARD)
    
    # rotate card back up
    if (w > h):
        return util.resize(np.rot90(warp), (SIZE_CARD_H, SIZE_CARD_H))

    return warp 

def get_approx_poly(card, do_rectify=True, image=None):
    perimeter = cv2.arcLength(card, True)

    approximated_poly = cv2.approxPolyDP(card, 0.1*perimeter, True)

    # TODO: deal with case where approximated_poly does not have 4 points (3 or 5)
    #kl = len(approximated_poly)
    #kif l != 4:
        #kfor p in approximated_poly:
            #kcv2.circle(image, (p[0][0], p[0][1]), 3, (255,0,0), 5)
        #kutil.show(image)
    #k# get rectified points in clockwise order

    if do_rectify:
        reapproximated_poly = util.rectify(approximated_poly)
        if reapproximated_poly.all():
            approximated_poly = reapproximated_poly
        else:
            print 'Not rectified!'
    return approximated_poly


def find_contours(bin_img, num=-1, return_area=False):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if num > 0:
        return contours[:num]

    return contours

def get_binary(img, thresh=150):
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
    if b != 0 and r/b > 2:
        return PROP_COLOR_RED

    # else, probably purple
    return PROP_COLOR_PURPLE 

def get_card_shape(card, training_set):
    binary = get_binary(card, thresh=150)
    contours = find_contours(binary)
    poly = get_approx_poly(contours[1], do_rectify=False)

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

def get_dropoff(array, maxratio=1.1):
    """Given array of values, return the index of the element where the ratio of each elem to the next drops off (assuming sorted input)"""

    ratios = np.divide(array, array[1:] + [1])

    count = 1

    for r in ratios:
        if r > maxratio:
            break
        else:
            count += 1

    return count

def get_card_number(card):
    binary = get_binary(card, thresh=150)
    #util.show(binary)
    contours = find_contours(binary)
    poly = get_approx_poly(contours[1], do_rectify=False)

    # forget about first outline of card
    contours_area = [cv2.contourArea(c) for c in contours][1:]

    return get_dropoff(contours_area, maxratio=1.1)

def get_card_texture(card, square=20):

    binary = get_binary(card, thresh=150)
    contour = find_contours(binary)[1]

    # get bounding rectangle
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    rect = cv2.getRectSubPix(card, (square,square), (x+w/2, y+h/2))

    gray_rect = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
    pixel_std = np.std(gray_rect)

    if pixel_std > 4.5:
        return PROP_TEXTURE_STRIPED

    elif np.mean(gray_rect) > 150:
        return PROP_TEXTURE_EMPTY

    else:
        return PROP_TEXTURE_SOLID



def test():
#    # 3 cards on flat table
    cards_3 = cv2.imread('images/set-3-texture.jpg')
    
    # 5 cards at an angle    
    cards_5_tilt = cv2.imread('images/set-5-random.jpg')
    

    thresh_3 = get_binary(cards_3)
    contours = find_contours(thresh_3, 3)

    assert len(transform_cards(cards_3, contours, 3)) == 3

    res5 = detect_cards(cards_5_tilt, 5)
    assert res5 is not None and len(res5) == 5 

    res3 = detect_cards(cards_3, 3)
    assert res3 is not None and len(res3) == 3

    for i in range(len(res5)):
        c = res5[i]
        # util.show(c, 'card')
        cv2.imwrite('images/cards/card-12-%d.jpg' % i, c)

    for i in range(len(res3)):
        c = res3[i]
        # util.show(c, 'card')
        cv2.imwrite('images/cards/card-3-%d.jpg' % i, c)

    # for cards detected, get properties
    for link in os.listdir('images/cards'):
        img = cv2.imread('images/cards/%s' % link)
        test_props(img)
    print 'tests pass'

def train():
    # train cards
    shape_diamond = cv2.imread('images/cards/card-5-4.jpg')
    shape_oblong = cv2.imread('images/cards/card-5-3.jpg')
    shape_squiggle = cv2.imread('images/cards/card-3-1.jpg')
    training_set = train_cards([shape_diamond, shape_oblong, shape_squiggle])
    return training_set

def test_props(img):
    color = PROP_COLOR_MAP[get_card_color(img)]
    shape = PROP_SHAPE_MAP[get_card_shape(img, train())]
    num =  get_card_number(img)
    texture =  PROP_TEXTURE_MAP[get_card_texture(img)]

    print '%d %s %s %s' % (num, color, shape, texture)
    util.show(img)
    # get_card_texture(img)
    print('---')

def test_bad_cards():
    # 3 of the 12 set that's bad
    cards_3_bad = cv2.imread('images/set-3-bad.jpg')
    thresh_3bad = get_binary(cards_3_bad)
    res3bad = detect_cards(cards_3_bad)
    assert res3bad is not None and len(res3bad) == 3
    
    # 12 cards
    cards_12 = cv2.imread('images/set-12-random.jpg')
    
    thresh_12bad = get_binary(cards_12, thresh=90)
    res12bad = detect_cards(cards_12)
    # util.show(cards_12)
    
    # Subset of 3, with the 1 problem card
    cards = res12bad
    for i in range(len(cards)):
        card = cards[i]
        test_props(card)
        cv2.imwrite('images/cards/card-12-%02d.jpg' % i, card)

    props = get_card_properties(res12bad, train())
    expected = [(3, 'RED', 'OBLONG'),(2, 'PURPLE', 'SQUIGGLE'),(1, 'GREEN', 'SQUIGGLE'),\
        (2, 'PURPLE', 'DIAMOND'), (2, 'GREEN', 'SQUIGGLE'), (1, 'GREEN', 'SQUIGGLE'), (2, 'RED', 'SQUIGGLE'), \
        (1, 'RED', 'DIAMOND'), (2, 'PURPLE', 'DIAMOND'), (1, 'RED', 'DIAMOND'), \
        (1, 'PURPLE', 'SQUIGGLE'), (1, 'GREEN', 'DIAMOND')]
    
    # assert props == expected

    pretty_print_properties(props)

    print 'tests pass'
