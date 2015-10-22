import set_solver as ss
import cv2 

# open video feed
# solve for every frame

WINDOW_NAME = 'video'
def main():
    cv2.namedWindow(WINDOW_NAME)
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = false

    while rval:
        # do set stuff
        cards = ss.detect_cards(frame, draw_rects=True)
        cv2.imshow(WINDOW_NAME, frame)
        rval, frame = vc.read()

        if(len(cards) < 1):
            continue

        # get property of cards and print
        props = ss.get_card_properties(cards)

        if len(props) > 0:
            ss.pretty_print_properties(props)
            print('----')


main()