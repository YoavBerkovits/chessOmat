import cv2
import numpy as np
import math
import copy
import gui_img_manager

# import filter_colors
# import chess_helper

"""
This file is responsible for getting an image and returning only the board's
image, projected to be rectangular.
"""

##### Image resize and cut dimensions #####
RESIZE_WIDTH = 600
RESIZE_HEIGHT = 600
CUT_UP = 70
CUT_DOWN = 1550

##### Line intersection filter #####
VER_MIN_INTERSECT = 6
HOR_MIN_INTERSECT = 6

##### Angle Ranges #####
VER_MIN_ANGLE = 0.9
VER_MAX_ANGLE = 2.3
VER_LEFT_MIN_ANGLE = 1.5
VER_LEFT_MAX_ANGLE = 2.2
VER_RIGHT_MIN_ANGLE = 1
VER_RIGHT_MAX_ANGLE = 1.5
HOR_MIN_ANGLE = 0.15
HOR_MAX_ANGLE = 3.0
HOR_DIFF_ANGLE = 0.09

##### General hough parameters #####
RHO_RES = 1
THETA_RES = np.pi / 180
MIN_VOTES = 115
MIN_LENGTH = 80
MAX_GAP = 30

##### Gaussian Threshold parameters #####
GAUSS_MAX_VALUE = 255
GAUSS_BLOCK_SIZE = 115
GAUSS_C = 13

##### Edge detection parameters #####
EDGE_DST = 10
EDGE_KSIZE = 3
EDGE_SCALE = 10

##### Finding pixels on lines #####
MAX_PIXEL_DISTANCE_FROM_LINE = 15

##### Finding fourth line #####
FOURTH_LINE_FIT_A = -0.02644
FOURTH_LINE_FIT_B = 26.37
FOURTH_LINE_FIT_C = 0.3048
FOURTH_LINE_FIT_D = 54.22
UPPER_RECT_H1 = (-150)  # -100
UPPER_RECT_H2 = (-70)
FOURTH_LINE_RHO_RES = 3
FOURTH_LINE_THETA_RES = np.pi / 180
FOURTH_LINE_MIN_VOTES = 125
FOURTH_LINE_MIN_LENGTH = 115
FOURTH_LINE_MAX_GAP = 30
PROJECTION_IMAGE_PADDING_RATIO = 1.0 / 7

DEBUG = False

class identify_board:

    def __init__(self):
        self.first = False

    """
    :return image of board, including an extra line above the board.
    """

    def get_board_image(self, img):
        fy_shrink = 500 / len(img)
        fx_shrink = 500 / len(img[0])
        resizeImg = cv2.resize(img, (len(img[0]), len(img)), fx=fx_shrink,
                               fy=fy_shrink)
        resizeImg = resizeImg[CUT_UP:CUT_DOWN, :]
        resizeImgGrey = cv2.cvtColor(resizeImg, cv2.COLOR_RGB2GRAY)
        # get lines from image, and edge-image
        edgeim = self.get_edge_image(resizeImgGrey)
        egdeim_copy = copy.deepcopy(edgeim)
        ver, her = self.amen_yaavod(edgeim)
        lines = self.filter_lines3(ver, her)
        # self.draw_lines(lines,egdeim_copy)
        points = self.get_point_for_rect_cut(lines)

        # find exectly the forth line
        croped_img, x_tikun, y_tikun = self.rect_cutter(egdeim_copy,
                                                        [points[0], points[1]])

        bottom_theta = self.get_theta(lines[1])

        # self.draw_lines([],croped_img)

        forth_line = self.find_specific_line(croped_img, bottom_theta, lines, x_tikun,
                                             y_tikun)
        # self.draw_lines(forth_line,croped_img)
        # get final points
        final_points = self.get_final_points(lines, forth_line, x_tikun, y_tikun)
        # self.draw_lines_by_points(final_points,egdeim_copy)
        board_img = self.projection(final_points, resizeImg)

        return board_img

    def find_lines2(self, img):
        linesP = cv2.HoughLinesP(img, RHO_RES, THETA_RES,
                                 MIN_VOTES, None, MIN_LENGTH, MAX_GAP)
        up_down = []
        left_right = []
        for line in linesP:
            for l in line:
                if self.get_theta(l) < VER_MAX_ANGLE and self.get_theta(
                        l) > VER_MIN_ANGLE:
                    up_down.append(l)
                if self.get_theta(l) < HOR_MIN_ANGLE or self.get_theta(
                        l) > HOR_MAX_ANGLE:  # WTF
                    left_right.append(l)

        # self.draw_lines(up_down,img)
        # self.draw_lines(left_right,img)
        return up_down, left_right

    def is_in_line(self, point, line):
        if line[0] >= point[0] - MAX_PIXEL_DISTANCE_FROM_LINE and line[2] <= point[0] + MAX_PIXEL_DISTANCE_FROM_LINE and \
                line[1] >= point[1] - MAX_PIXEL_DISTANCE_FROM_LINE and line[3] <= point[
            1] + MAX_PIXEL_DISTANCE_FROM_LINE:
            return True
        if line[0] >= point[0] - MAX_PIXEL_DISTANCE_FROM_LINE and line[2] <= point[0] + MAX_PIXEL_DISTANCE_FROM_LINE and \
                line[1] <= point[1] + MAX_PIXEL_DISTANCE_FROM_LINE and line[3] >= point[
            1] - MAX_PIXEL_DISTANCE_FROM_LINE:
            return True
        if line[0] <= point[0] + MAX_PIXEL_DISTANCE_FROM_LINE and line[2] >= point[0] - MAX_PIXEL_DISTANCE_FROM_LINE and \
                line[1] >= point[1] - MAX_PIXEL_DISTANCE_FROM_LINE and line[3] <= point[
            1] + MAX_PIXEL_DISTANCE_FROM_LINE:
            return True
        if line[0] <= point[0] + MAX_PIXEL_DISTANCE_FROM_LINE and line[2] >= point[0] - MAX_PIXEL_DISTANCE_FROM_LINE and \
                line[1] <= point[1] + MAX_PIXEL_DISTANCE_FROM_LINE and line[3] >= point[
            1] - MAX_PIXEL_DISTANCE_FROM_LINE:
            return True
        return False

    def num_of_cutting(self, line, lines):
        count = 0
        for l in lines:
            point = self.get_cutoff_point(line, l)
            if self.is_in_line(point, l) and self.is_in_line(point, line):
                count += 1
        return count

    def amen_yaavod(self, img):
        ver, hor = self.find_lines2(img)
        rank_ver = [self.num_of_cutting(item, hor) for item in ver]
        rank_hor = [self.num_of_cutting(item, ver) for item in hor]
        for i in range(len(rank_ver)):
            i2 = len(rank_ver) - 1 - i
            if (rank_ver[i2] < VER_MIN_INTERSECT):
                ver = ver[:i2] + ver[i2 + 1:]
        for i in range(len(rank_hor)):
            i2 = len(rank_hor) - 1 - i
            if (rank_hor[i2] < HOR_MIN_INTERSECT):
                hor = hor[:i2] + hor[i2 + 1:]
        #        self.draw_lines(ver,img)
        #        self.draw_lines(hor,img)
        return ver, hor

    def get_lines_theta(self, lines):
        new_lines = []
        for i in range(len(lines)):
            current = lines[i]
            theta = self.get_theta(current)
            new_lines.append(theta)
        return new_lines

    def gausThresholdChess(self, img):
        gaus = cv2.adaptiveThreshold(img, GAUSS_MAX_VALUE,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, GAUSS_BLOCK_SIZE, GAUSS_C)
        # cv2.imshow("saharur",gaus)
        # k =  cv2.waitKey(0)
        return gaus

    def edgeDetectionChess(self, img):
        lapalacian = cv2.Laplacian(img, cv2.CV_64F, EDGE_DST, EDGE_KSIZE, EDGE_SCALE)

        return lapalacian

    def get_theta(self, line):
        if line[2] == line[0]:
            theta = math.pi / 2
        else:
            theta = (float)(math.atan2(float(line[3] - line[1]), float(line[2] - line[0])))
        return (theta % math.pi);

    def get_cutoff_point(self, line1, line2):
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]
        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]
        if x1 == x2:
            m1 = 10000000000
        else:
            m1 = float(float(y2 - y1) / float(x2 - x1))
        if x3 == x4:
            m2 = 10000000000
        else:
            m2 = float(float(y4 - y3) / float(x4 - x3))
        n1 = y1 - m1 * x1
        n2 = y3 - m2 * x3

        x = int((n1 - n2) / (m2 - m1))
        y = int(m1 * x + n1)
        point = []
        point.append(x)
        point.append(y)
        return point

    def get_length(self, lines):
        [x1, y1] = self.get_cutoff_point(lines[0], lines[1])
        [x2, y2] = self.get_cutoff_point(lines[1], lines[2])
        length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return length

    def find_avg_line_color(self, line, img):
        dx = 2
        dy = 2
        x_min = min(line[0], line[2]) - dx
        x_max = max(line[0], line[2]) + dx
        y_min = min(line[1], line[3]) - dy
        y_max = max(line[1], line[3]) + dy
        color = []
        color.append(0)
        color.append(0)
        color.append(0)
        if x_max > len(img[0]) or x_min < 0 or y_max > len(img) or y_min < 0:
            return [500, 500, 500]
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                color[0] += img[j][i][0]
                color[1] += img[j][i][1]
                color[2] += img[j][i][2]
        color[0] = (color[0] / (x_max - x_min)) / (y_max - y_min)
        color[1] = (color[1] / (x_max - x_min)) / (y_max - y_min)
        color[2] = (color[2] / (x_max - x_min)) / (y_max - y_min)
        return color

    '''
    def remove_lines_colors(self,lines, real_img):
        real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
        chessh = chess_helper.chess_helper(chess_helper.chess_helper.ME)
        width = len(real_img[0])
        height = len(real_img)
        cut_im = real_img[int(4 * height / 10):int(4 * height / 5), int(5 * width / 10):int(7 * width / 10)]
        #    cv2.imshow('s', cut_im)
        #    cv2.waitKey(0)
        colorf = filter_colors.filter_colors(cut_im, chessh)
        colors = filter_colors.filter_colors.get_main_colors(colorf, cut_im)

        range = 18
        new_lines = []
        for line in lines:
            added = False
            color2 = self.find_avg_line_color(line, real_img)
            for color in colors:
                if filter_colors.filter_colors.color_dist(colorf, color2, color) < range and added == False:
                    added = True
                    new_lines.append(line)
        return new_lines
    '''

    def get_real_theta_left(self, line1, line2):
        theta1 = self.get_theta(line1)
        theta2 = self.get_theta(line2)
        final_theta = (theta1 - theta2) % (math.pi)
        return final_theta

    def get_real_theta_right(self, line1, line2):
        theta1 = self.get_theta(line1)
        theta2 = self.get_theta(line2)
        final_theta = math.pi - (theta2 - theta1) % (math.pi)
        return final_theta

    def get_distance(self, theta1, theta2, length):
        ## last hatama
        # a=0.2423
        # b=2.522
        # c=0.5098
        # d=-6.509
        a = FOURTH_LINE_FIT_A
        b = FOURTH_LINE_FIT_B
        c = FOURTH_LINE_FIT_C
        d = FOURTH_LINE_FIT_D
        if theta1 > math.pi / 2:
            theta1 = math.pi - theta1
        if theta2 > math.pi / 2:
            theta2 = math.pi - theta2
        distance = length * (a * math.atan(b * (theta1 + theta2) + d) + c)
        return distance

    def find_rect_locaition(self, line1, line2, distance):
        theta = self.get_theta(line2)
        x1 = line1[0] + distance * math.cos(theta)
        y1 = line1[1] - distance * math.sin(theta)
        x2 = line1[2] + distance * math.cos(theta)
        y2 = line1[3] - distance * math.sin(theta)

        return [x1, y1, x2, y2]

    def get_point_for_rect_cut(self, lines):

        length = self.get_length(lines)
        theta1 = self.get_real_theta_left(lines[0], lines[1])
        theta2 = self.get_real_theta_right(lines[1], lines[2])
        distance = self.get_distance(theta1, theta2, length)
        new_line = self.find_rect_locaition(lines[1], lines[0], distance)
        points = []
        point1 = self.get_cutoff_point(lines[0], new_line)
        point2 = self.get_cutoff_point(lines[2], new_line)
        points.append(point1)
        points.append(point2)

        return points

    def fix_points_for_projection(self, points_lst):
        fix_point = 20
        p1 = points_lst[0]
        p1[0] -= fix_point
        p1[1] -= fix_point
        p2 = points_lst[1]
        p2[0] += fix_point
        p2[1] -= fix_point
        p3 = points_lst[2]
        p3[0] -= fix_point
        p3[1] += fix_point
        p4 = points_lst[3]
        p4[0] += fix_point
        p4[1] += fix_point

    def projection(self, pointslst, img, width, height):
        pts1 = np.float32(pointslst)
        pts2 = np.float32([[int(width * PROJECTION_IMAGE_PADDING_RATIO),
                            int(height * PROJECTION_IMAGE_PADDING_RATIO)],
                           [int(width * (1 - PROJECTION_IMAGE_PADDING_RATIO)),
                            int(height * PROJECTION_IMAGE_PADDING_RATIO)],
                           [int(width * PROJECTION_IMAGE_PADDING_RATIO),
                            int(height * (1 - PROJECTION_IMAGE_PADDING_RATIO))],
                           [int(width * (1 - PROJECTION_IMAGE_PADDING_RATIO)),
                            int(height * (1 - PROJECTION_IMAGE_PADDING_RATIO))]])
        #        self.fix_points_for_projection(pts1)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (width, height))
        # cv2.imshow("ss",dst)
        # k = cv2.waitKey(0)
        return dst

    def rect_cutter(self, img, points):
        #        print math.cos(theta)
        h2 = UPPER_RECT_H2
        h1 = int(UPPER_RECT_H1)
        x1 = points[0][0]
        x2 = points[1][0]
        y1 = min(points[0][1], points[1][1])
        y2 = max(points[0][1], points[1][1])
        crop_img = img[y1 + h1:y2 + h2, x1:x2]

        x_tikun = min(x1, x2)
        y_tikun = y1 + h1

        return crop_img, x_tikun, y_tikun

    def filter_lines3(self, verticel_lines_lst, horizontal_lines_lst):
        left_lines_lst = []
        right_lines_lst = []
        final_lines = []

        for line in verticel_lines_lst:
            theta = self.get_theta(line)
            if theta > VER_LEFT_MIN_ANGLE and theta < VER_LEFT_MAX_ANGLE:
                left_lines_lst.append(line)
            if theta > VER_RIGHT_MIN_ANGLE and theta < VER_RIGHT_MAX_ANGLE:
                right_lines_lst.append(line)

        min_x = left_lines_lst[0]
        for line in left_lines_lst:
            y1 = min(line[1], line[3])
            y2 = min(min_x[1], min_x[3])
            y = max(y1, y2)
            m1, n1 = self.find_m_n(line)
            x_line = float(float(y - n1) / float(m1))
            m2, n2 = self.find_m_n(min_x)
            x_min = float((float(y - n2)) / float(m2))
            if x_line < x_min:
                min_x = line
        final_lines.append(min_x)

        max_y = horizontal_lines_lst[0]
        for line in horizontal_lines_lst:
            x1 = min(line[0], line[2])
            x2 = min(max_y[0], max_y[2])
            x = max(x1, x2)
            m1, n1 = self.find_m_n(line)
            y_line = x * m1 + n1
            m2, n2 = self.find_m_n(max_y)
            y_max = x * m2 + n2
            if y_line > y_max:
                max_y = line
        final_lines.append(max_y)

        max_x = right_lines_lst[0]
        for line in right_lines_lst:
            y1 = min(line[1], line[3])
            y2 = min(max_x[1], max_x[3])
            y = max(y1, y2)
            m1, n1 = self.find_m_n(line)
            x_line = (y - n1) / m1
            m2, n2 = self.find_m_n(max_x)
            x_max = (y - n2) / m2
            if x_line > x_max:
                max_x = line
        final_lines.append(max_x)

        return final_lines

    def find_m_n(self, line):
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        if x1 == x2:
            m1 = 10000000000
        else:
            m1 = float(float(y2 - y1) / float(x2 - x1))
        n1 = y1 - m1 * x1

        return m1, n1

    """
    returns perpendicular line that bisects given line.
    """

    def get_perpendicular(self, line):
        midx = (line[0] + line[2]) * 1.0 / 2
        midy = (line[1] + line[3]) * 1.0 / 2
        if (line[1] == line[3]):
            m_inv = 1000000000
        else:
            m_inv = (line[0] - line[2]) * 1.0 / (line[1] - line[3])
        if (m_inv == 0):
            m_inv = -10000000000
        else:
            m_inv = -1.0 / m_inv
        topy = 0
        topx = midx - midy * m_inv
        return [int(midx), int(midy), int(topx), int(topy)]

    def find_specific_line(self, croped_img, bottom_theta, threelines,
                           x_tikun, y_tikun):
        croped_img = cv2.convertScaleAbs(croped_img)
        linesP = cv2.HoughLinesP(croped_img, FOURTH_LINE_RHO_RES, FOURTH_LINE_THETA_RES,
                                 FOURTH_LINE_MIN_VOTES, None,
                                 FOURTH_LINE_MIN_LENGTH, FOURTH_LINE_MAX_GAP)
        l = []
        for line in linesP:
            l.append(line[0])

        #        self.draw_lines(l,croped_img)
        lines = []
        for line in linesP:
            theta = self.get_theta(line[0]) - bottom_theta
            if abs(theta) < HOR_DIFF_ANGLE:
                lines.append(line)
        x1, y1 = self.get_cutoff_point(threelines[0], threelines[1])
        x2, y2 = self.get_cutoff_point(threelines[2], threelines[1])
        line = [x1, y1, x2, y2]
        ver_line = self.get_perpendicular(line)
        ver_line = [ver_line[0] - x_tikun, ver_line[1] - y_tikun, ver_line[
            2] - x_tikun, ver_line[3] - y_tikun]
        #        self.draw_lines([ver_line],croped_img)
        min_y = lines[0]
        #        print(ver_line)
        #        print(min_y)
        min_cut_y = (self.get_cutoff_point(min_y[0], ver_line))[1]
        for ln in lines:
            cut = self.get_cutoff_point(ln[0], ver_line)
            if cut[1] < min_cut_y:
                min_y = ln
                min_cut_y = cut[1]

        return min_y

    def draw_lines_by_points(self, points, img):
        line0 = points[0] + points[1]
        line1 = points[2] + points[0]
        line2 = points[1] + points[3]
        line3 = points[3] + points[2]
        new_lines = []
        new_lines.append(line0)
        new_lines.append(line1)
        new_lines.append(line2)
        new_lines.append(line3)
        bin = copy.deepcopy(img)
        bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

        for i in range(0, len(new_lines)):
            l = new_lines[i]
            cv2.line(bin, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow("ophir hamelech", bin)
        k = cv2.waitKey(0)


    def get_line_image(self, points, img):
        line0 = points[0] + points[1]
        line1 = points[2] + points[0]
        line2 = points[1] + points[3]
        line3 = points[3] + points[2]
        new_lines = []
        new_lines.append(line0)
        new_lines.append(line1)
        new_lines.append(line2)
        new_lines.append(line3)
        bin = copy.deepcopy(img)
        bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

        for i in range(0, len(new_lines)):
            l = new_lines[i]
            cv2.line(bin, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
        return bin


    def get_image_from_img(self, image, should_cut):
        real_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        fy_shrink = 500 / len(img)
        fx_shrink = 500 / len(img[0])
        if should_cut:
            resizeImg = cv2.resize(img, (len(img[0]), len(img)), fx=fx_shrink,
                                   fy=fy_shrink)
            resizeImg = resizeImg[CUT_UP:CUT_DOWN, :]
            real_img = cv2.resize(real_img, (len(img[0]), len(img)), fx=fx_shrink,
                                  fy=fy_shrink)
            real_img \
                = real_img[CUT_UP:CUT_DOWN, :]
        else:
            resizeImg = img

        # cv2.imshow('ss', resizeImg)
        # cv2.waitKey(0)
        '''
        points = []
        p1 = [int(0.05 * len(resizeImg[0])), 0]
        p2 = [int(0.95*len(resizeImg)), 0]
        p3 = [0, len(resizeImg)]
        p4 = [len(resizeImg[0]), len(resizeImg)]
        points.append(p1)
        points.append(p2)
        points.append(p3)
        points.append(p4)
        resizeImg = self.projection(points, resizeImg,len(resizeImg[0]),
                                    len(resizeImg) )

        #cv2.imshow('sss', resizeImg)
        #cv2.waitKey(0)
        '''
        threshim = self.gausThresholdChess(resizeImg)
        edgeim = self.edgeDetectionChess(threshim)
        edgeim = cv2.convertScaleAbs(edgeim)
        if(DEBUG):
            cv2.imshow("sahar", edgeim)
            cv2.waitKey(0)
        return edgeim, real_img

    def get_image_from_filename(self, imgFileName, should_cut):
        real_img = cv2.imread(imgFileName, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        fy_shrink = 500 / len(img)
        fx_shrink = 500 / len(img[0])
        if should_cut:
            resizeImg = cv2.resize(img, (len(img[0]), len(img)), fx=fx_shrink,
                                   fy=fy_shrink)
            resizeImg = resizeImg[CUT_UP:CUT_DOWN, :]
            real_img = cv2.resize(real_img, (len(img[0]), len(img)), fx=fx_shrink,
                                  fy=fy_shrink)
            real_img \
                = real_img[CUT_UP:CUT_DOWN, :]
        else:
            resizeImg = img

        # cv2.imshow('ss', resizeImg)
        # cv2.waitKey(0)
        '''
        points = []
        p1 = [int(0.05 * len(resizeImg[0])), 0]
        p2 = [int(0.95*len(resizeImg)), 0]
        p3 = [0, len(resizeImg)]
        p4 = [len(resizeImg[0]), len(resizeImg)]
        points.append(p1)
        points.append(p2)
        points.append(p3)
        points.append(p4)
        resizeImg = self.projection(points, resizeImg,len(resizeImg[0]),
                                    len(resizeImg) )

        #cv2.imshow('sss', resizeImg)
        #cv2.waitKey(0)
        '''
        threshim = self.gausThresholdChess(resizeImg)
        edgeim = self.edgeDetectionChess(threshim)
        edgeim = cv2.convertScaleAbs(edgeim)
        return edgeim, real_img

    def get_edge_image(self, img):
        threshim = self.gausThresholdChess(img)
        edgeim = self.edgeDetectionChess(threshim)
        edgeim = cv2.convertScaleAbs(edgeim)
        return edgeim

    def draw_lines(self, lineslst, img):
        new_img = copy.deepcopy(img)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
        for i in range(0, len(lineslst)):
            l = lineslst[i]
            cv2.line(new_img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
            #        img = cv2.resize(img, (0, 0), fx=1, fy=1)
        cv2.imshow('ophir', new_img)
        k = cv2.waitKey(0)
        return

    def get_final_points(self, lines, fourth_line, x_tikun, y_tikun):
        # print(fourth_line)
        real_forth_line = [fourth_line[0][0] + x_tikun, fourth_line[0][1] + y_tikun, fourth_line[0][2] + x_tikun,
                           fourth_line[0][3] + y_tikun]

        final_points = []
        final_points.append(self.get_cutoff_point(real_forth_line, lines[0]))
        final_points.append(self.get_cutoff_point(real_forth_line, lines[2]))
        final_points.append(self.get_cutoff_point(lines[1], lines[0]))
        final_points.append(self.get_cutoff_point(lines[1], lines[2]))
        return final_points

    def test(self, foldername):
        # get lines from image, and edge-image
        for j in range(0, 44):
            try:
                edgeim, real_img = self.get_image_from_filename(
                    foldername + "/" + str(j) + '.jpg', True)
                egdeim_copy = copy.deepcopy(edgeim)
                ver, her = self.amen_yaavod(edgeim)
                self.draw_lines(ver,egdeim_copy)
                self.draw_lines(her,egdeim_copy)

                lines = self.filter_lines3(ver, her)
                self.draw_lines(lines,egdeim_copy)

                points = self.get_point_for_rect_cut(lines)

                # find exectly the forth line
                croped_img, x_tikun, y_tikun = self.rect_cutter(egdeim_copy,
                                                                [points[0], points[1]])
                #               self.draw_lines([],croped_img)

                forth_line = self.find_specific_line(croped_img,
                                                     self.get_theta(lines[1]),
                                                     lines, x_tikun, y_tikun)
                #                self.draw_lines(forth_line,croped_img)

                # get final points
                final_points = self.get_final_points(lines, forth_line, x_tikun, y_tikun)

                #                self.draw_lines_by_points(final_points,egdeim_copy)
                img = self.projection(final_points, real_img, RESIZE_WIDTH,
                                      RESIZE_HEIGHT)
                if(DEBUG):
                    cv2.imshow('sss',img)
                    cv2.waitKey(0)
                #cv2.imwrite(foldername + '/projected/' + str(j) + '.jpg', img)
                # print (j)
            except:
                print(str(j) + " failed")

    def main(self, img):

        edgeim, real_img = self.get_image_from_img(img, True)
        
        gui_img_manager.add_img(edgeim)

        try:
            egdeim_copy = copy.deepcopy(edgeim)
            ver, her = self.amen_yaavod(edgeim)
            if(DEBUG):
                self.draw_lines(ver,edgeim)
                self.draw_lines(her,edgeim)
            lines = self.filter_lines3(ver, her)
            points = self.get_point_for_rect_cut(lines)

            # find exectly the forth line
            croped_img, x_tikun, y_tikun = self.rect_cutter(egdeim_copy,
                                                            [points[0], points[1]])

            forth_line = self.find_specific_line(croped_img,
                                                 self.get_theta(lines[1]),
                                                 lines, x_tikun, y_tikun)
            # get final points
            final_points = self.get_final_points(lines, forth_line, x_tikun,
                                                 y_tikun)
            #                self.draw_lines_by_points(final_points,egdeim_copy)
            gui_img_manager.add_img(self.get_line_image(final_points,edgeim))
            img = self.projection(final_points, real_img, RESIZE_WIDTH,
                                  RESIZE_HEIGHT)
            edgeim = self.projection(final_points, edgeim, RESIZE_WIDTH,
                                  RESIZE_HEIGHT)
            gui_img_manager.add_img(img)
            return img, edgeim
        except:
            print("identify board has failed")
            return real_img, edgeim

# a = identify_board()
# img = cv2.imread("images/0.jpg", cv2.IMREAD_COLOR)
# new = a.main(img)


# if you want to see the image:

# new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
# a.draw_lines([],new)
