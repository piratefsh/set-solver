import cv2
import sys
import numpy as np 

def canny(img):
    edges = cv2.Canny(img, threshold1=200, threshold2=40)

    return edges 

def hough(edges, output, threshold):

    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho 
        y0 = b * rho 

        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))

        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(output, (x1,y1), (x2,y2), (0,0,255), 1)
    return

def hough_p(edges, output, threshold, min_line_length, max_line_gap):

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold, \
        min_line_length, max_line_gap)

    if lines is not None and len(lines) > 0:
        return lines[0]
    return []

def show(img, window_name):
    cv2.imshow(window_name, img)
    return window_name

def destroy(window_name):
    cv2.destroyWindow(window_name)

def test(t, ml, mg):
    destroy('canny')
    # filename = 'images/brooklyn-bridge.jpg'
    filename = 'images/shapes.png'

    img = cv2.imread(filename, 0)
    print(img)
    color_img = cv2.imread(filename)
    canny_edges = canny(img)
    c = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)

    hough(canny_edges, c, t)
    lines, l_sum, l_avg = hough_p(canny_edges, c, t, ml, mg)
    print 'threshold: %d, min_line_len, %d, max_gap: %d\n%d lines found' \
        % (t, ml, mg, lines)
    print 'sum is %d' % l_sum
    print 'avg is %d' % l_avg

    show(c, 'canny')

def get_video(smoothing=5):

    lines_queue = []

    cv2.namedWindow('preview')
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = false

    while rval:
        key = cv2.waitKey(20)
        if key == 27:
            break

        canny_edges = canny(frame)

        f = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        
        lines_queue.append(hough_p(canny_edges, f, 150, 10, 30))

        # while we have less than 5 sets of lines in buffer, move on
        if len(lines_queue) < smoothing:
            continue
        else:

            # take last five frames
            #lastn = lines_queue[-smoothing:]

            # draw all lines
            for lines in lines_queue:
                for x1,y1,x2,y2 in lines:
                    cv2.line(f, (x1,y1), (x2,y2), (0,0,255), 2)

            lines_queue.pop(0)


        cv2.imshow('preview', f)
        rval, frame = vc.read()

    cv2.destroyWindow('preview')
    vc.release()


