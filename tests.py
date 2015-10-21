from set_solver import * 
from set_test import game 

def test():
    # 3 cards on flat table
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
    print('---')

def test_main():
    # 3 of the 12 set that's bad
    cards_3_bad = cv2.imread('images/set-3-bad.jpg')
    thresh_3bad = get_binary(cards_3_bad)
    res3bad = detect_cards(cards_3_bad)
    assert res3bad is not None and len(res3bad) == 3
    
    # 12 cards
    cards_12 = cv2.imread('images/set-12-random-90deg.jpg')
    
    thresh_12bad = get_binary(cards_12, thresh=90)
    res12bad = detect_cards(cards_12)
    # util.show(cards_12)
    
    # Subset of 3, with the 1 problem card
    cards = res12bad
    for i in range(len(cards)):
        card = cards[i]
        # test_props(card)
        cv2.imwrite('images/cards/card-12-%02d.jpg' % i, card)

    props = get_card_properties(res12bad, train())
    pretty_print_properties(props)

    g = game(cards=props)
    sets = g.play(prnt=True)
    if sets:
        pretty_print_properties(sets)
    else:
        print 'no sets :('



    # assert props == expected


    print 'tests pass'
