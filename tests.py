import cv2, util, random
import set_solver as s
import cv2.cv as cv
from set_test import game
from collections import Counter
import code

def test():
    # 3 cards on flat table
    cards_3 = cv2.imread('images/set-3-texture.jpg')

    # 5 cards at an angle
    cards_5_tilt = cv2.imread('images/set-5-random.jpg')


    thresh_3 = s.get_binary(cards_3)
    contours = s.find_contours(thresh_3, 3)

    assert len(s.transform_cards(cards_3, contours, 3)) == 3

    res5 = s.detect_cards(cards_5_tilt, 5)
    assert res5 is not None and len(res5) == 5 

    res3 = s.detect_cards(cards_3, 3)
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
    shape_diamond = cv2.imread('images/training/diamond.jpg')
    shape_oblong = cv2.imread('images/training/oblong.jpg')
    shape_squiggle = cv2.imread('images/training/squiggle.jpg')
    training_set = s.train_cards([shape_diamond, shape_oblong, shape_squiggle])
    return training_set

def test_props(img):
    color = s.PROP_COLOR_MAP[s.get_card_color(img)]
    shape = s.PROP_SHAPE_MAP[s.get_card_shape(img, train())]
    num =  s.get_card_number(img)
    texture =  s.PROP_TEXTURE_MAP[s.get_card_texture(img)]

    print '%d %s %s %s' % (num, color, shape, texture)
    util.show(img)
    print('---')

def main():
    # 3 of the 12 set that's bad
    cards_3_bad = cv2.imread('images/set-3-bad.jpg')
    thresh_3bad = s.get_binary(cards_3_bad)
    res3bad = s.detect_cards(cards_3_bad)
    assert res3bad is not None and len(res3bad) == 3

    # 12 cards
    cards_12 = cv2.imread('images/set-12-random-2sets.jpg')

    thresh_12bad = s.get_binary(cards_12)
    res12bad = s.detect_cards(cards_12, draw_rects=False)
    util.show(cards_12)

    # Subset of 3, with the 1 problem card
    cards = res12bad
    for i in range(len(cards)):
        card = cards[i]
        # test_props(card)
        cv2.imwrite('images/cards/card-12-%02d.jpg' % i, card)

    props = s.get_card_properties(res12bad, train())
    s.pretty_print_properties(props)

    g = game(cards=props)
    sets = g.play(prnt=True)
    if sets:
        for st in sets:
            s.pretty_print_properties(st)
            print('---')
    else:
        print 'no sets :('

    print 'tests pass'


def play_game(file_in, printall=False, draw_contours=True, \
              resize_contours=True, draw_rects=False, \
              sets_or_no=False):
    """Takes in an image file, finds all sets, and pretty prints them to screen.
    if printall - prints the identities of all cards in the image
    if draw_contours - outlines the cards belonging to each set
    if resize_contours - enlarges contours for cards belonging to multiple sets to avoid overlay
    if draw_rects - draws box rects around cards belonging to each set
    if sets_or_no - outlines the image in green or red, depending on whether there are any sets present"""
    orig_img = cv2.imread(file_in)
    img = s.resize_image(orig_img, 900)

    contours, detected = s.detect_cards(img, draw_rects=False, \
                                        return_contours=True)
    props = s.get_card_properties(detected, train())

    if printall:
        s.pretty_print_properties(props)

    g = game(cards=props)
    sets = g.play(idx=True)

    # RED, ORANGE, YELLOW, GREEN, BLUE, //INDIGO,// VIOLET
    BGR_RAINBOW = [ (0, 0, 255), (0, 127, 255), (0, 255, 255), \
                    (0, 255, 0), (255, 0, 0), #(130, 0, 75),
                    (255, 0, 139) ]

    if sets:
        # choose a group of colors at random to represent the set of sets
        #COLORS = random.sample(BGR_RAINBOW, len(sets))
        COLORS = random_color_palette(len(sets))

        if resize_contours:
            # count number of sets that each winning card index belongs to
            counter = Counter( card[0] for st in sets for card in st )

        for i, st in enumerate(sets):
            color = COLORS[i]
            st_indices, st_props = zip(*st)
            s.pretty_print_properties(st_props)
            print ('---')

            if draw_contours or draw_rects:
                winning_contours = [ contours[c] for c in st_indices ]

                if draw_contours:
                    if resize_contours:
                        for idx in st_indices:
                            # set base thickness
                            thickness = 3
                            count = counter[idx]
                            if count > 1:
                                thickness += 6*counter[idx]
                            counter[idx] -= 1
                            cv2.drawContours(img, contours, idx, color, thickness)
                            #cv2.drawContours(img, contours[idx]*3, -1, color, 3)

                    else:
                        cv2.drawContours(img, winning_contours, -1, color, 3)

                if draw_rects:
                    # get bounding rectangles
                    winning_rects = [ cv2.minAreaRect(c) for c in winning_contours ]
                    for rect in winning_rects:
                        # convert to ints
                        r = [ (int(x), int(y)) for x,y in cv.BoxPoints(rect) ]
                        cv2.rectangle(img, r[0], r[2], color)

    else:
        print 'no sets :('

    if sets_or_no:
        height, width, _ = img.shape
        BORDER_SCALAR = 0.01
        border_h, border_w = ( int(dim*BORDER_SCALAR) for dim in (height, width))

        # indices 0 or 1 correspond to bool for if no sets (BGR for red) or sets (green)
        BORDER_COLORS = [ (19,19,214), (94,214,19) ]

        img_outlined = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w,\
                                          cv2.BORDER_CONSTANT, value = BORDER_COLORS[bool(sets)])
        util.show(img_outlined)

    else:
        util.show(img)


# inspired by http://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
def random_color_palette(n, BGR=True):
    """Generates a random, aesthetically pleasing set of n colors (list of BGR tuples - because opencv is silly - if BGR; else HSV)"""
    SATURATION = 0.6
    VALUE = 0.95
    GOLDEN_RATIO_INVERSE = 0.618033988749895

    # see: https://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB
    def hsv2bgr(hsv):
        h, s, v = hsv
        # compute chroma
        c = v*s
        h_prime = h*6.0
        x = c*( 1 - abs(h_prime %2 - 1) )
        if h_prime >= 5: rgb = (c,0,x)
        elif h_prime >= 4: rgb = (x,0,c)
        elif h_prime >= 3: rgb = (0,x,c)
        elif h_prime >= 2: rgb = (0,c,x)
        elif h_prime >= 1: rgb = (x,c,0)
        else: rgb = (c,x,0)
        m = v-c
        rgb = tuple( 255.0*(val+m) for val in rgb )
        # flip tuple to return (B,G,R)
        return rgb[::-1]

    # random float in [0.0, 1.0)
    hue = random.random()
    l_hues = [hue]

    for i in xrange(n-1):
        # generate evenly distributed hues by random walk using the golden ratio!
        # (mod 1, to stay within hue space)
        hue += GOLDEN_RATIO_INVERSE
        hue %= 1
        l_hues.append(hue)

    if not BGR:
        return [ (h, SATURATION, VALUE) for h in l_hues ]

    return [ hsv2bgr((h, SATURATION, VALUE)) for h in l_hues ]
