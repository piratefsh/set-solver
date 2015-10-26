import cv2, sys, os
import cv2.cv as cv
import sys
import numpy as np
import util as util
import os
import code
import set_constants as sc

def resize_image(img, new_width=600):
    """Given cv2 image object and maximum dimension, returns resized image such that height or width (whichever is larger) == max dimension"""
    h, w, _ = img.shape
    new_height = int((1.0*h/w)*new_width)
    resize = np.zeros((new_width, new_height))
    print (new_width, new_height)
    resized = cv2.resize(img, (new_width, new_height))

    return resized

def get_card_properties(cards, training_set=None):
    if training_set == None:
        training_set = get_training_set()
    properties = []
    for img in cards:
        num = get_card_number(img)
        color = get_card_color(img)
        shape = get_card_shape(img, training_set)
        texture = get_card_texture(img)
        p = (num, color, shape, texture)
        if None not in p:
            properties.append(p) 
    return properties

def pretty_print_properties(properties):
    for p in properties:
        num, color, shape, texture = p
        print '%d %s %s %s' % (num, sc.PROP_COLOR_MAP[color],
                               sc.PROP_SHAPE_MAP[shape],
                               sc.PROP_TEXTURE_MAP[texture])

def detect_cards(img, draw_rects=False, return_contours=False):
    if img is None:
        return None

    img_binary = get_binary(img)
    contours = find_contours(img_binary)
    num_cards = get_dropoff([cv2.contourArea(c)
                             for c in contours], maxratio=1.5)
    cards = transform_cards(img, contours, num_cards, draw_rects=draw_rects)
    transformed_cards = transform_cards(img, contours, num_cards, draw_rects=draw_rects)

    if return_contours:
        print 'contours'
        return (contours, transformed_cards)
    else:
        print 'no contours'
        return (transformed_cards)

def transform_cards(img, contours, num, draw_rects=False):
    cards = []
    for i in xrange(num):
        if i > len(contours) - 1:
            continue
        card = contours[i]

        # get bounding rectangle
        rect = cv2.minAreaRect(card)
        r = cv.BoxPoints(rect)

        # convert to ints
        r = [(int(x), int(y)) for x, y in r]

        if draw_rects:
            cv2.rectangle(img, r[0], r[2], sc.COLOR_RED)

        try:
            transformed = transform_card(card, img)
        except:
            # print 'Error processing card!! :o'
            continue
        
        if transformed is not None:
            cards.append(transformed)

    return cards


def transform_card(card, image):
    # find out if card is rotated
    x, y, w, h = cv2.boundingRect(card)
    card_shape = [[0, 0], [sc.SIZE_CARD_W, 0], [
        sc.SIZE_CARD_W, sc.SIZE_CARD_H], [0, sc.SIZE_CARD_H]]

    # get poly of contour
    approximated_poly = get_approx_poly(card, do_rectify=True)

    if approximated_poly is None:
        # could not find card poly 
        return None

    dest = np.array(card_shape, np.float32)

    # do transformatiom
    transformation = cv2.getPerspectiveTransform(approximated_poly, dest)
    warp = cv2.warpPerspective(image, transformation, sc.SIZE_CARD)

    # rotate card back up
    if (w > h):
        return util.resize(np.rot90(warp), (sc.SIZE_CARD_H, sc.SIZE_CARD_W))

    return warp

def get_approx_poly(card, do_rectify=True, image=None):
    perimeter = cv2.arcLength(card, True)

    approximated_poly = cv2.approxPolyDP(card, 0.1*perimeter, True)

    # TODO: deal with case where approximated_poly does not have 4 points (3 or 5)
    #kl = len(approximated_poly)
    # kif l != 4:
    # kfor p in approximated_poly:
    #kcv2.circle(image, (p[0][0], p[0][1]), 3, (255,0,0), 5)
    # kutil.show(image)
    # k# get rectified points in clockwise order

    if do_rectify:
        reapproximated_poly = util.rectify(approximated_poly)
        if reapproximated_poly.all():
            approximated_poly = reapproximated_poly
        else:
            #print 'Not rectified!'
            return None
    return approximated_poly


def find_contours(bin_img, num=-1, return_area=False):
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if num > 0:
        return contours[:num]

    return contours


def get_binary(img, thresh=150):
    preprocessed = util.preprocess(img)
    _, threshold = cv2.threshold(
        preprocessed, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    return threshold


def get_canny(img):
    preprocessed = util.preprocess(img)
    canny = cv2.Canny(preprocessed, threshold1=200, threshold2=50)
    dilated = cv2.dilate(canny, (10, 10))
    return dilated


def get_card_color(card):
    card = get_shape_only(card)

    if card is None:
        return None 
        
    blue = [pix[0] for row in card for pix in row]
    green = [pix[1] for row in card for pix in row]
    red = [pix[2] for row in card for pix in row]

    bgr = (min(blue), min(green), min(red))
    b, g, r = bgr
    # if mostly green
    if max(bgr) == g:
        return sc.PROP_COLOR_GREEN

    # if a lot more red than blue, is probably red
    if b != 0 and r/b > 2:
        return sc.PROP_COLOR_RED

    # else, probably purple
    return sc.PROP_COLOR_PURPLE


def get_card_shape(card, training_set, thresh=150):
    # binary = get_binary(card, thresh=thresh)
    binary = get_canny(card)
    contours = find_contours(binary)

    # if canny doesn't give enough contours, fallback to binary
    if len(contours) < 2:
        binary = get_binary(card)
        contours = find_contours(binary)
        # if still not enough contours, consider invalid
        if len(contours) < 2:
            return None

    poly = get_approx_poly(contours[1], do_rectify=False)

    # for each card in trainings set, find one with most similarity
    diffs = []
    this_shape = get_shape_contour(card)
    for i, that_shape in training_set.items():

        # resize image
        this_shape_res = util.resize(this_shape, that_shape.shape)

        # find diff and its sum
        d = cv2.absdiff(this_shape_res, that_shape)
        sum_diff = np.sum(d)

        diffs.append(sum_diff)

    # return index of shape that has minimum difference
    return diffs.index(min(diffs)) + 1

# get bounding rect coords
def get_shape_bounding_rect(img):
    binary = get_canny(img)
    contours = find_contours(binary)

    # if canny doesn't give enough contours, fallback to binary
    if len(contours) < 2:
        binary = get_binary(img)
        contours = find_contours(binary)
        if len(contours) < 2:
            return None

    shape_contour = contours[1]
    x, y, w, h = cv2.boundingRect(shape_contour)
    return (y, y+h, x, x+w, contours)

# cropped out contour of shape
def get_shape_contour(img):
    rect = get_shape_bounding_rect(img)

    if rect is None:
        return None 

    y1, y2, x1, x2, contours = rect
    shape_img = util.draw_contour(contours, 1)
    cropped = shape_img[y1:y2, x1:x2]
    return cropped

# cropped out image of shape
def get_shape_only(img):
    rect = get_shape_bounding_rect(img)

    if rect is None:
        return None 
        
    y1, y2, x1, x2, _ = rect
    cropped = img[y1:y2, x1:x2]
    return cropped

def get_training_set():
    # train cards
    shape_diamond = cv2.imread('images/training/diamond.jpg')
    shape_oblong = cv2.imread('images/training/oblong.jpg')
    shape_squiggle = cv2.imread('images/training/squiggle.jpg')
    training_set = do_training([shape_diamond, shape_oblong, shape_squiggle])
    return training_set


def do_training(imgs):
    # train for shapes, return contours of shapes
    training_set = {}
    for i in range(len(imgs)):
        img = imgs[i]
        shape = get_shape_contour(img)
        training_set[i] = shape
    return training_set


def get_dropoff(array, maxratio=1.1):
    """
    Given array of values, return the index of the element 
    where the ratio of each elem to the next drops off (assuming sorted input)
    """

    array = [e for e in array if e > 0]

    ratios = np.divide(array, array[1:] + [1])

    count = 1

    for r in ratios:
        if r > maxratio:
            break
        else:
            count += 1

    return count


def get_card_number(card):
    binary = get_binary(card, thresh=180)
    contours = find_contours(binary)

    if len(contours) < 2:
        return None 

    poly = get_approx_poly(contours[1], do_rectify=False)

    # forget about first outline of card
    contours_area = [cv2.contourArea(c) for c in contours][1:]

    return get_dropoff(contours_area, maxratio=1.1)


def get_card_texture(card, square=20):

    binary = get_binary(card, thresh=150)
    contours = find_contours(binary)

    if len(contours) < 2:
        return None 
    
    contour = contours[1]

    # get bounding rectangle
    rect = cv2.boundingRect(contour)
    x, y, w, h = rect

    rect = cv2.getRectSubPix(card, (square, square), (x+w/2, y+h/2))

    gray_rect = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
    pixel_std = np.std(gray_rect)

    if pixel_std > 4.5:
        return sc.PROP_TEXTURE_STRIPED

    elif np.mean(gray_rect) > 150:
        return sc.PROP_TEXTURE_EMPTY

    else:
        return sc.PROP_TEXTURE_SOLID
