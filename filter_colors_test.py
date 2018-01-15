import scipy
import scipy.misc
import scipy.cluster
import chess_helper
from scipy import misc
import filter_colors
import time
import cv2
import os

BLACK = (0.0, 0.0, 0.0)
BLACK_SHOW = (0.0, 0.0, 0.0)
WHITE_SHOW = (1.0, 1.0, 1.0)
MY_TURN = True
SOURCE = True
PIXELS = (160, 160)


def cmpT(t1, t2):
    return sorted(t1) == sorted(t2)


def color_dist(color1, color2):
    light1 = max(color1) / 2 + min(color1) / 2
    light2 = max(color2) / 2 + min(color2) / 2
    return abs(light1 - light2)


"""
:return an image with 4/3/2 colors only for test.
"""


def catalogue_colors_show(im, is_white, main_colors, me_him_colors):
    main_colors_white = main_colors[1:]
    main_colors_black = [main_colors[0]] + main_colors[2:]
    if is_white:
        cat_im = fit_colors_show(im, main_colors_white, is_white, me_him_colors)
    else:
        cat_im = fit_colors_show(im, main_colors_black, is_white, me_him_colors)
    return cat_im


"""
    :return image fit to 4 main colors.
    """


def fit_colors_show(im, main_colors, is_white, me_him_colors):
    me_show = me_him_colors[0]
    him_show = me_him_colors[1]
    new_im = []
    for rowidx in range(len(im)):
        i = 0
        row = im[rowidx]
        new_im.append([])
        for pix in row:
            min_dist = color_dist(pix, main_colors[0])
            if is_white:
                new_im[rowidx].append(WHITE_SHOW)
                if not cmpT(main_colors[1], main_colors[0]):
                    dist = color_dist(pix, main_colors[1])
                    if dist < min_dist:
                        min_dist = dist
                        new_im[rowidx][i] = me_show
                if not cmpT(main_colors[2], main_colors[0]):
                    dist = color_dist(pix, main_colors[2])
                    if dist < min_dist:
                        new_im[rowidx][i] = him_show
            else:
                new_im[rowidx].append(BLACK_SHOW)
                if not cmpT(main_colors[1], main_colors[0]):
                    dist = color_dist(pix, main_colors[1])
                    if dist < min_dist:
                        min_dist = dist
                        new_im[rowidx][i] = me_show
                if not cmpT(main_colors[2], main_colors[1]):
                    dist = color_dist(pix, main_colors[2])
                    if dist < min_dist:
                        new_im[rowidx][i] = him_show
            i += 1
    return new_im


def update_colors_show(main_colors):
    me_show = (0.6, 0.8, 0.8)
    him_show = (0.5, 0.2, 0.2)
    if cmpT(main_colors[2], main_colors[0]):
        me_show = BLACK_SHOW
    elif cmpT(main_colors[2], main_colors[1]):
        me_show = WHITE_SHOW
    if cmpT(main_colors[3], main_colors[0]):
        him_show = BLACK_SHOW
    elif cmpT(main_colors[3], main_colors[1]):
        him_show = WHITE_SHOW
    return (me_show, him_show)


"""
gets main_im (for the main colors), and 2 ims of the same square in 2
different moves
image.
"""


def tester(main_im_name, im1_name, im2_name, is_my_turn, loc, is_source):
    main_im = misc.imresize(misc.imread(main_im_name), (600, 600))
    if is_my_turn:
        chesshelper = chess_helper.chess_helper(chess_helper.chess_helper.ME)
    else:
        chesshelper = chess_helper.chess_helper(chess_helper.chess_helper.RIVAL)
    colorfilter = filter_colors.filter_colors(main_im, chesshelper)
    im1 = misc.imread(im1_name)
    # im1 = misc.imresize(misc.imread(im1_name), PIXELS)
    colorfilter.set_prev_im(im1)
    #im1_square = colorfilter.get_square_image(im1, loc, is_my_turn)
    im2 = misc.imread(im2_name)
    # im2 = misc.imresize(misc.imread(im2_name), PIXELS)
    #im2_square = colorfilter.get_square_image(im2, loc, is_my_turn)
    #is_white = chesshelper.square_color(loc)
    #main_colors = colorfilter.main_colors
    #me_him_colors = update_colors_show(main_colors)
    #im1_cat_show = catalogue_colors_show(im1_square, is_white, main_colors, me_him_colors)
    #im2_cat_show = catalogue_colors_show(im2_square, is_white, main_colors, me_him_colors)
    # scipy.cv2.imwrite('tester_1.JPEG', im1_square)
    # scipy.cv2.imwrite('tester_2.JPEG', im2_square)
    # scipy.cv2.imwrite('tester_1_cat.JPEG', im1_cat_show)
    # scipy.cv2.imwrite('tester_2_cat.JPEG', im2_cat_show)
    #t_init = time.time()
    square_diff = colorfilter.get_square_diff(im2, loc, is_source)
    #t_final = time.time()
    scipy.cv2.imwrite('test_dir/tester_diff_' + loc + '.jpg', square_diff)
    #tdelta = t_final - t_init
    #print('measured time is: ', round(tdelta, 3), ' sec')
    return


def tester2(im2_name, loc, is_source,
            colorfilter):
    im2 = cv2.resize(cv2.imread(im2_name), (PIXELS, PIXELS))
    square_diff = colorfilter.get_square_diff(im2, loc, is_source)
    cv2.imwrite('test_dir/tester_diff.jpg', square_diff)
    return


def tester2_1(main_colors, im1_name, im2_name, is_my_turn, loc, is_source):
    if is_my_turn:
        chesshelper = chess_helper.chess_helper(chess_helper.chess_helper.ME)
    else:
        chesshelper = chess_helper.chess_helper(
            chess_helper.chess_helper.RIVAL)
    colorfilter = filter_colors.filter_colors(im1_name, chesshelper, main_colors)
    im1 = misc.imresize(misc.imread(im1_name), (PIXELS, PIXELS))
    colorfilter.set_prev_im(im1)
    tester2(im2_name, loc, is_source, colorfilter)
    return


def tester3(init_im, im1, im2, is_my_turn, is_source):
    for letter in range(ord('f'), ord('i')):
        for num in range(8):
            tester(init_im, im1, im2, is_my_turn, chr(letter) + str(num + 1), is_source)


#tester3('3.jpg', '3.jpg', '4.jpg', MY_TURN, not SOURCE)
"""
tester2_1([[104.97780915, 36.93843029, 11.57384154],
         [204.86528027, 171.79534614, 96.93486107],
         [166.37594893, 125.71247412, 42.53088337],
         [15.20291907, 8.09617689, 4.42914715]]
        , '1l.jpg', '2l.jpg', MY_TURN, 'f3', not SOURCE)
"""
