import cv2
import sys
import numpy as np 

def canny(img):
    edges = cv2.Canny(img, threshold1=200, threshold2=100)

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

    for x1,y1,x2,y2 in lines[0]:
        cv2.line(output, (x1,y1), (x2,y2), (0,255,0), 1)

    calc = lambda line: ((line[0] - line[2])**2 + (line[1] - line[3])**2)**.5
    lines_sum = sum([calc(line) for line in lines[0]])

    lines_avg = np.mean([calc(line) for line in lines[0]])

    return len(lines[0]), lines_sum, lines_avg

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

def get_video():

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
        
        hough_p(canny_edges, f, 150, 10, 10)

        cv2.imshow('preview', f)
        rval, frame = vc.read()

    cv2.destroyWindow('preview')
    vc.release()


