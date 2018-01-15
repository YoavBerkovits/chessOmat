import math
import cv2
import identify_board
import copy
import numpy as np
import bisect
import gui_img_manager

### DEBUG FLAG ###
DEBUG = False


MIN_GRID_SIZE = 1.0 / 15
MAX_GRID_SIZE = 1.0 / 7
MAX_LINE_DIST_RATIO = 1.0 / 15
RESIZE_HEIGHT = identify_board.RESIZE_HEIGHT
RESIZE_WIDTH = identify_board.RESIZE_WIDTH
MAX_NUM_LINES = 11
MAX_LINES_IN_GRID = 9
LINE_SERIES_COEFF = 5
LINE_SERIES_POWER = 1
ANGLE_STD_COEFF = 0.8
MIN_NUM_ANGLES_AVG = 3

### Filter lines by surrounding color parameters ###
MIN_MIXED_AREA_VAR_LIGHTNESS = 15
MIXED_AREA = 0
UNIFORM_AREA = 1
INSIDE_LINE =  0
OUTSIDE_LINE = 1
BAD_LINE = 2

LINE_METRIC_ANGLE_ERROR_COEFF = 150
LINE_METRIC_DIST_ERROR_COEFF = 6

LINE_UNIFORM_SCANNER_PIXEL_JUMP = 15

#####sizes of the window in the convolution
PartOfWindowHeight = 25
PartOfWindowLength = 40
ConvSkipX = 4
ConvSkipY = 4
#####number of changes to be an legal image
CHANGE_COLOR_SAF = 4

##### Gaussian Threshold parameters #####
GAUSS_MAX_VALUE = 255
GAUSS_BLOCK_SIZE = 115
GAUSS_C = 13

class board_cut_fixer:

    def get_theta(self,line):
        try:
            if line[2] == line[0]:
                theta = math.pi / 2
            else:
                theta = (float)(math.atan2(float(line[3] - line[1]), float(line[2] - line[0])))
        except:
            print('1'
                  '')
        return (theta % math.pi);

        """"""

    def draw_points(self, img, points):
        if(DEBUG):
            img = copy.deepcopy(img)
            for point in points:
                fix_point = (int(point[0]), int(point[1]))
                cv2.circle(img, fix_point, 10, [0, 255, 255])
            cv2.imshow('image', img)
            cv2.waitKey(0)

    """
    rounded modulo
    """

    def modulo(self, x, y):

        mod = x % y
        if mod > y / 2:
            mod = y - mod
        return mod

    def get_board_limits(self, im, points):
        board_size = len(im)
        best_pair = self.get_grid_origin_pair(points, board_size)
        p1 = best_pair[0]
        p2 = best_pair[1]
        self.draw_points(im,best_pair)
        axes= self.get_axes_from_pair(best_pair)
        x_axis = axes[0]
        y_axis = axes[1]
        row_scores = {}
        col_scores = {}
        for i in range(-MAX_NUM_LINES, MAX_NUM_LINES+1):
            row_scores[i] = 0
            col_scores[i] = 0
        for p in points:
            grid_loc = self.get_point_grid_location(p1, x_axis, y_axis, p)
            score = self.get_point_grid_metric(grid_loc)
            row = round(grid_loc[0])
            col = round(grid_loc[1])
            row_scores[row] = row_scores[row] + score
            col_scores[col] = col_scores[col] + score

        max_row_score = 0
        max_col_score = 0
        left_row_idx = -12
        top_row_idx = -12
        for i in range(-MAX_NUM_LINES, 1):
            row_score = (row_scores[i]) ** 0.5 + (row_scores[i + 8]) ** 0.5
            col_score = (col_scores[i]) ** 0.5 + (col_scores[i + 8]) ** 0.5
            if (row_score > max_row_score):
                max_row_score = row_score
                left_row_idx = i
            if (col_score > max_col_score):
                max_col_score = col_score
                top_col_idx = i

        ul_pt = (p1[0] + left_row_idx * x_axis[0] + top_col_idx * y_axis[0],
                 p1[1] + left_row_idx * x_axis[1] + top_col_idx * y_axis[1])

        br_pt = (p1[0] + (left_row_idx + 8) * x_axis[0] + (top_col_idx + 8) * y_axis[0],
                 p1[1] + (left_row_idx + 8) * x_axis[1] + (top_col_idx + 8) * y_axis[1])

        ur_pt = (p1[0] + (left_row_idx+8) * x_axis[0] + top_col_idx * y_axis[0],
                 p1[1] + (left_row_idx+8) * x_axis[1] + top_col_idx * y_axis[1])

        bl_pt = (p1[0] + left_row_idx * x_axis[0] + (top_col_idx + 8) * y_axis[0],
                 p1[1] + left_row_idx * x_axis[1] + (top_col_idx + 8) * y_axis[1])
        return (ul_pt, ur_pt, br_pt, bl_pt)

    def Make_3d_List_2_2d_List(self, list):
        list2d = []
        for li in list:
            for l in li:
                list2d.append(l)
        return list2d

    def connectLines(self, lines, board_size):
        max_dis = board_size * MAX_LINE_DIST_RATIO
        theta = self.get_theta(lines[0])
        new_lines = []
        values = []
        new_lines.append(lines[0])
        if theta < 0.03 or theta > 3.1:
            values.append(lines[0][1])
            for i in range(1, len(lines)):
                line = lines[i]
                added = True
                for j in range(len(values)):
                    if abs(line[1] - values[j]) < 20:
                        added = False
                if added:
                    new_lines.append(line)
                    values.append(line[1])
        if theta < (math.pi / 2) + 0.03 or theta > (math.pi / 2) - 0.03:
            values.append(lines[0][0])
            for i in range(1, len(lines)):
                line = lines[i]
                added = True
                for j in range(len(values)):
                    if abs(line[0] - values[j]) < max_dis:
                        added = False
                if added:
                    new_lines.append(line)
                    values.append(line[0])
        return new_lines

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

    """
    fixes projected image
    """

    def projection(self, pointslst, img, frame):
        pts1 = np.float32(pointslst)
        x_hi = frame[1]*RESIZE_WIDTH/8
        x_lo = frame[3]*RESIZE_WIDTH/8
        y_hi = (8 - frame[0]) * RESIZE_HEIGHT / 8
        y_lo = (8-frame[2])*RESIZE_HEIGHT/8
        pts2 = np.float32([[x_lo, y_hi], [x_hi, y_hi],
                           [x_hi, y_lo],[x_lo, y_lo]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(img, M, (RESIZE_WIDTH, RESIZE_HEIGHT))
        if(DEBUG):
            cv2.imshow("image",dst)
            k = cv2.waitKey(0)
        return dst

    def draw_lines(self, lineslst, img):
        if(DEBUG):
            new_img = copy.deepcopy(img)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
            for i in range(0, len(lineslst)):
                l = lineslst[i]
                cv2.line(new_img, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
            #        img = cv2.resize(img, (0, 0), fx=1, fy=1)
            cv2.imshow('image', new_img)
            k = cv2.waitKey(0)
            return

    def get_lines(self,img):
        linesP = cv2.HoughLinesP(img, 1, math.pi / 180, 100, minLineLength=100, maxLineGap=30)
        lines = self.Make_3d_List_2_2d_List(linesP)
        hor = []
        ver = []
        for l in lines:
            if self.get_theta(l) < (math.pi / 2) + 0.1 and self.get_theta(l) > (math.pi / 2) - 0.1:
                ver.append(l)
            if self.get_theta(l) < 0.18 or self.get_theta(l) > 2.96:
                hor.append(l)
    #        hor.sort(key=lambda x: x[1], reverse=True)
        #hor = self.connectLines(hor,len(img))
        #ver = self.connectLines(ver,len(img))
        return hor,ver




    """
    given a list of lines, and a function that gives a value for each line,
    find the most accurate item in the series, and the delta value of the
    series.
    :return a list of best fits to the series, SORTED BY FIT!
    """
    def get_line_series(self,lines, valfunc, lower_d, upper_d, num_vals):
        lines.sort(key = valfunc)
        vals = [valfunc(l) for l in lines]
        best_d = 0
        best_score = 0
        best_lines = []
        best_line_index = 0
        for i in range(len(vals)):
            for j in range(i+1,len(vals)):
                if not i==j: # delta val cannot be 0.
                    d = abs(vals[j]-vals[i])
                    if(lower_d<=d<=upper_d):
                        start = -(num_vals-1)
                        end = num_vals-1
                        scores = []
                        tmp_lines = []
                        for n in range(start,end+1):

                            val_n = vals[i]+n*d
                            if(val_n<-d or val_n>RESIZE_WIDTH+d): #exit if
                                # beyond bounds
                                scores.append(0)
                                tmp_lines.append(lines[i])
                            else:
                                val_closest = min(vals, key=lambda x:abs(x-val_n))
                                # find closest value <= val_n
                                val_closest_idx = bisect.bisect_right(vals,
                                                                      val_n)-1
                                if (val_closest_idx+1)<len(vals) and vals[val_closest_idx+1]-val_n < \
                                                val_n-vals[val_closest_idx]:
                                    val_closest_idx = val_closest_idx+1
                                val_closest = vals[val_closest_idx]
                                line_closest = lines[val_closest_idx]
                                diff = abs(val_n - val_closest)
                                if diff<=d/2:
                                    scores.append(1/(
                                    1+LINE_SERIES_COEFF*diff*1.0/d)**LINE_SERIES_POWER)
                                    tmp_lines.append(line_closest)
                                else:
                                    scores.append(0)
                                    tmp_lines.append(lines[i])
                                # score it
                                # will
                                #  have :(

                        score = sum(scores[0:num_vals])
                        best_window_score = score
                        offset = 0
                        for k in range(0,len(scores)-num_vals):
                            score = score + scores[num_vals+k]-scores[k]
                            if score>best_window_score:
                                best_window_score = score
                                offset = k+1

                        if best_window_score>best_score:
                            best_d = d
                            best_score = best_window_score
                            best_lines = tmp_lines[offset:offset+num_vals]
                            best_line_index = num_vals-offset-1
        if(DEBUG):
            print("best linear series:")
            print("d/boardsize = " + str(best_d/RESIZE_WIDTH))
            print("score: "+ str(best_score))
        return best_lines, best_line_index, best_d

    def line_eq(self, l1,l2):
        return l1[0]==l2[0] and l1[1] == l2[1] and l1[2]==l2[2] and l1[3]==l2[3]

    def get_last_line_extrapolation(self, lines, baseline, line_num, get_theta, get_pos, d):

        tmplines = []
        foundbase = False
        for line in lines:
            if not (foundbase and self.line_eq(baseline, line)):
                tmplines.append(line)
            if self.line_eq(baseline, line):
                foundbase = True

        angles = [get_theta(l) for l in tmplines]
        angle_avg = sum(angles) / len(angles)
        angle_std = np.std(angles)
        if(DEBUG):
            print("angles:" + str(angles))
            print("angles std:" + str(angle_std))
            print("angles avg:" + str(angle_avg))
        angles2 = [angle for angle in angles if abs(angle - angle_avg) < ANGLE_STD_COEFF * angle_std]
        if len(angles2) < MIN_NUM_ANGLES_AVG:
            angles2 = sorted(angles, key=lambda x:abs(x-angle_avg))
            angles2 = angles2[:MIN_NUM_ANGLES_AVG]
        angles = angles2
        if(DEBUG):
            print("after fix ")
            print("angles:" + str(angles))
            print("angles avg:" + str(angle_avg))
        angle_avg = sum(angles) / len(angles)
        origpos = get_pos(baseline)
        positions = [get_pos(l) for l in tmplines]
        origloc = (origpos - min(positions)) * 1.0 / d
        finalLoc = -origloc+line_num
        finalPos = origpos + finalLoc * d
        return finalPos , angle_avg

    def make_hor_line(self, point, angle):
        x1 = point[0]
        y1 = point[1]
        x2 = RESIZE_WIDTH - 1
        y2 = math.atan(angle) * (x2 - x1) + y1
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return [x1,y1,x2,y2]

    def make_ver_line(self, point, angle):
        x1 = point[0]
        y1 = point[1]
        y2 = RESIZE_HEIGHT - 1
        x2 = math.atan(-angle+math.pi/2) * (y2 - y1) + x1
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return [x1, y1, x2, y2]

    def get_board_limits(self,up_line, right_line, bot_line, left_line):
        points = []
        points.append(self.get_cutoff_point(up_line,left_line))
        points.append(self.get_cutoff_point(up_line,right_line))
        points.append(self.get_cutoff_point(bot_line,right_line))
        points.append(self.get_cutoff_point(bot_line,left_line))
        return points

    """
    check if line's area contains black&white.
    """
    def get_area_type(self,line, img, above):
        dx = 5
        dy = 5
        if(above):
            x_min = max(min(line[0], line[2]) - dx,0)
            x_max = min(max(line[0], line[2]) + dx,len(img[0]))
            y_min = max(min(line[1], line[3]),0)
            y_max = min(max(line[1], line[3]) + dy,len(img))
        else:
            x_min = max(min(line[0], line[2]) - dx, 0)
            x_max = min(max(line[0], line[2]) + dx, len(img[0]))
            y_min = max(min(line[1], line[3]) - dy, 0)
            y_max = min(max(line[1], line[3]), len(img))
        avg_lightness = 0
        for i in range(x_min, x_max, LINE_UNIFORM_SCANNER_PIXEL_JUMP):
            for j in range(y_min, y_max):
                avg_lightness += max(img[j][i]) / 2 + min(img[j][i]) / 2

        avg_lightness = avg_lightness*1.0/(len(range(x_min,x_max,
                                                     LINE_UNIFORM_SCANNER_PIXEL_JUMP))*(y_max-y_min))
        var_lightness = 0
        for i in range(x_min, x_max, LINE_UNIFORM_SCANNER_PIXEL_JUMP):
            for j in range(y_min, y_max):
                var_lightness += abs(max(img[j][i]) / 2 + min(img[j][i]) / 2 - avg_lightness)
        var_lightness= var_lightness*1.0/(len(range(x_min,x_max,LINE_UNIFORM_SCANNER_PIXEL_JUMP))*(
            y_max-y_min))
        if var_lightness>MIN_MIXED_AREA_VAR_LIGHTNESS:
            return MIXED_AREA
        return UNIFORM_AREA

    """
    color img plz
    """
    def get_lines_types(self,lines, img):
        types = [(self.get_area_type(line,img,True),self.get_area_type(line,img,False)) for line in lines]
        bad = []
        out = []
        ins = []
        for i in range(len(types)):
            if types[i][0] == MIXED_AREA and types[i][1] == MIXED_AREA:
                ins.append(lines[i])
            elif types[i][0] == MIXED_AREA and types[i][1] == UNIFORM_AREA:
                out.append(lines[i])
            elif types[i][0] == UNIFORM_AREA and types[i][1] == UNIFORM_AREA:
                bad.append(lines[i])
        return bad,out,ins

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

    def gausThresholdChess(self, img):
        gaus = cv2.adaptiveThreshold(img, GAUSS_MAX_VALUE,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, GAUSS_BLOCK_SIZE,
                                     GAUSS_C)
#        cv2.imshow("raveh",gaus)
#        cv2.waitKey(0)
        # cv2.imshow("saharur",gaus)
        # k =  cv2.waitKey(0)
        return gaus

    def doConv(self, img, line):
        window_length = len(img) // PartOfWindowLength
        window_width = len(img[0]) // PartOfWindowHeight

        changeCounter = 0
        lastcolorFlag = 0  ##zero if last is white, 1 if black
        newim = copy.deepcopy(img)
        for i in range(len(img) - window_length - 1):
            whitepix = 0
            blackpix = 0
            m, n = self.find_m_n(line)
            for j in range(i, i + window_length, ConvSkipX):
                for k in range(int(m) * j + int(n) - window_width,
                               int(m) * j + int(n), ConvSkipY):
                    if k>=len(newim) or j>=len(newim[0]):
                        break
                    if (newim[k][j] > 0):
                        whitepix = whitepix + 1
                    else:
                        blackpix = blackpix + 1
           #         newim[k][j] = 0
          #  cv2.imshow("raveh", newim)
          #  cv2.waitKey(0)
                        #                newim[k][j] = 0
                        #        cv2.imshow("n",newim)
                        #        cv2.waitKey(0)
            # print(whitepix,blackpix)
            if whitepix > blackpix:
                specificcolorFlag = 1  ##zero if last is white, 1 if black
            else:
                specificcolorFlag = 0
            if specificcolorFlag != lastcolorFlag:
                changeCounter = changeCounter + 1
                lastcolorFlag = specificcolorFlag
        return changeCounter

    def is_proj_correct(self, img, line):
        newimg = self.gausThresholdChess(img)
        resize_image = cv2.resize(newimg, (0, 0), fx=1, fy=1)
        kernel = np.ones((5, 5), np.uint8)
        dilim = cv2.dilate(resize_image, kernel)
        changeColor = self.doConv(dilim, line)
        if (changeColor > CHANGE_COLOR_SAF):
            return True
        return False

    def remove_bad_bottom_lines(self, hor_lines, valfunc, real_img):
        img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        hor = sorted(hor_lines, key=lambda x:-valfunc(x))
        for i in range(len(hor)):
            if self.is_proj_correct(img,hor[i]):
                return hor[i:]
        print ("all lines are bad")
        return hor
    """
    finds highest reasonable hor line, and returns it's index
    """
    def get_highest_horizontal_line_index(self, lines, base_idx, d,
                                          get_pos, get_theta):

        ##### Find average angle, without weird lines that are errors. #####
        tmplines = []
        baseline = lines[base_idx]

        ## remove baseline duplicates
        for line in lines:
            if not self.line_eq(baseline, line):
                tmplines.append(line)
        tmplines.append(baseline)

        ## get avg angle
        print(tmplines)
        angles = [get_theta(l) for l in tmplines]
        angle_avg = sum(angles) / len(angles)
        angle_std = np.std(angles)
        if (DEBUG):
            print("angles:" + str(angles))
            print("angles std:" + str(angle_std))
            print("angles avg:" + str(angle_avg))
        angles2 = [angle for angle in angles if
                   abs(angle - angle_avg) < ANGLE_STD_COEFF * angle_std]
        if len(angles2) < MIN_NUM_ANGLES_AVG:
            angles2 = sorted(angles, key=lambda x: abs(x - angle_avg))
            angles2 = angles2[:MIN_NUM_ANGLES_AVG]
        angles = angles2
        if (DEBUG):
            print("after fix ")
            print("angles:" + str(angles))
            print("angles avg:" + str(angle_avg))
        angle_avg = sum(angles) / len(angles)

        ##### Find highest line that is close enough to such angle #####
        def line_metric(line, idx):
            score = (8-idx)-LINE_METRIC_ANGLE_ERROR_COEFF*abs(get_theta(
                line)-angle_avg) - LINE_METRIC_DIST_ERROR_COEFF*self.modulo(
                abs(get_pos(line)-get_pos(baseline)),d)*1.0/d
            return score

        scores = [(i,line_metric(lines[i],i)) for i in range(9) if not (
            self.line_eq(lines[i], baseline) and (not i==base_idx))]
        best_line_idx = scores[(max(range(len(scores)), key=lambda x:(scores[
                                                                         x])[1]))][0]
        line_metric(lines[best_line_idx],best_line_idx)
        if(self.line_eq(lines[best_line_idx],baseline)): # index is not
        # real. fix it by finding
            best_line_idx = base_idx

        return best_line_idx
    """
      finds highest reasonable hor line, and returns it's index
      """

    def get_best_vertical_line_pair_index(self, lines, base_idx, d,
                                          get_pos, get_theta):

        ##### Find average angle, without weird lines that are errors. #####
        tmplines = []
        baseline = lines[base_idx]

        ## remove baseline duplicates
        for line in lines:
            if not self.line_eq(baseline, line):
                tmplines.append(line)
        tmplines.append(baseline)

        ## get avg angle
        print(tmplines)
        angles = [get_theta(l) for l in tmplines]
        angle_avg = sum(angles) / len(angles)
        angle_std = np.std(angles)
        if (DEBUG):
            print("angles:" + str(angles))
            print("angles std:" + str(angle_std))
            print("angles avg:" + str(angle_avg))
        angles2 = [angle for angle in angles if
                   abs(angle - angle_avg) < ANGLE_STD_COEFF * angle_std]
        if len(angles2) < MIN_NUM_ANGLES_AVG:
            angles2 = sorted(angles, key=lambda x: abs(x - angle_avg))
            angles2 = angles2[:MIN_NUM_ANGLES_AVG]
        angles = angles2
        if (DEBUG):
            print("after fix ")
            print("angles:" + str(angles))
            print("angles avg:" + str(angle_avg))
        angle_avg = sum(angles) / len(angles)

        ##### Find highest line that is close enough to such angle #####
        def pair_metric(line, idx, line2, idx2):
            score = abs(idx2-idx) - LINE_METRIC_ANGLE_ERROR_COEFF * (abs(
                get_theta(line) - angle_avg) + abs(
                get_theta(line2) - angle_avg)) - LINE_METRIC_DIST_ERROR_COEFF*(self.modulo(
                abs(get_pos(line)-get_pos(baseline)),d)+self.modulo(
                abs(get_pos(line2)-get_pos(baseline)),d))*1.0/d
            return score

        scores = [(i, j, pair_metric(lines[i], i, lines[j], j)) for i in
                   range(9) if not (self.line_eq(lines[i], baseline) and (
                not i == base_idx)) for j in range(9) if not (self.line_eq(
            lines[j], baseline) and (
                not j == base_idx)) ]
        best_pair_indices = scores[(max(range(len(scores)), key=lambda x: (
            scores[x])[2]))][0:2]
        pair_metric(lines[best_pair_indices[0]], best_pair_indices[0],
                    lines[best_pair_indices[1]], best_pair_indices[1])
        if (self.line_eq(lines[best_pair_indices[0]], baseline)):  # index is not
            # real. fix it by finding
            best_pair_indices = (base_idx ,best_pair_indices[1])

        if (self.line_eq(lines[best_pair_indices[1]], baseline)):  # index
            # is not
            # real. fix it by finding
            best_pair_indices = (best_pair_indices[0],base_idx)

        return best_pair_indices

    def main(self, real_img, edgeim):

        def get_theta(line):
            if line[2] == line[0]:
                return math.pi / 2
            else:
                return (float)(math.atan2(float(line[3] - line[1]),
                                           float(line[2] - line[0]))) % math.pi

            """"""

        """
         do not use with vertical lines.
         """

        def get_y_point_on_line(line):
            x1 = line[0]
            y1 = line[1]
            y = y1 - (x1 - RESIZE_WIDTH // 2) * (line[3] - y1) * 1.0 / (line[2] - x1)
            return y

        """
        do not use with horizontal lines.
        """

        def get_x_point_on_line(line):
            x1 = line[0]
            y1 = line[1]
            return x1 - (y1 - RESIZE_HEIGHT // 2) * (line[2] - x1) * 1.0 / (line[3]  - y1)

        def get_theta_hor(line):
            angle = self.get_theta(line)
            if angle > math.pi / 2:
                return angle - math.pi
            return angle

        def get_theta_ver(line):
            return self.get_theta(line)
        try:
            hor, ver = self.get_lines(edgeim)

            hor = self.remove_bad_bottom_lines(hor,get_y_point_on_line,real_img)

            new_hor, best_hor_idx, hor_d = self.get_line_series(hor,
                                                             get_y_point_on_line, len(edgeim) * MIN_GRID_SIZE,
                                                             len(edgeim) * MAX_GRID_SIZE, 9)
            new_ver, best_ver_idx, ver_d = self.get_line_series(ver,
                                                             get_x_point_on_line, len(edgeim[0]) * MIN_GRID_SIZE,
                                                             len(edgeim[0]) * MAX_GRID_SIZE, 9)
            if(DEBUG):
                self.draw_lines(ver, edgeim)
                self.draw_lines(new_ver, edgeim)
                self.draw_lines(hor, edgeim)
                self.draw_lines(new_hor, edgeim)


            best_hor_idx = self.get_highest_horizontal_line_index(new_hor,
                                                                  best_hor_idx,
                                                                  hor_d,
                                                                  get_y_point_on_line, get_theta_hor)

            best_ver_pair = self.get_best_vertical_line_pair_index(new_ver,
                                                                  best_ver_idx,
                                                                  ver_d,
                                                                  get_x_point_on_line,
                                                                  get_theta_ver)

            line_num = 8-best_hor_idx
            left_num = min(best_ver_pair)
            right_num = max(best_ver_pair)
            up_line = new_hor[best_hor_idx]
            down_line = new_hor[8]
            left_line = new_ver[left_num]
            right_line = new_ver[right_num]
            """

            up_pos, up_angle = self.get_last_line_extrapolation(new_hor, best_hor, 0, get_theta_hor, lambda x: -get_y_point_on_line(x),
                                                                 hor_d)
            down_pos, down_angle = self.get_last_line_extrapolation(new_hor, best_hor, 8, get_theta_hor,
                                                                    lambda x: get_y_point_on_line(x), hor_d)
            left_pos, left_angle = self.get_last_line_extrapolation(new_ver, best_ver, 0, get_theta_ver,
                                                                    get_x_point_on_line, ver_d)
            right_pos, right_angle = self.get_last_line_extrapolation(new_ver, best_ver, 8, get_theta_ver,
                                                                     get_x_point_on_line, ver_d)
            up_point = (RESIZE_WIDTH * 1.0 / 2, -up_pos)
            down_point = (RESIZE_WIDTH * 1.0 / 2, -down_pos)
            left_point = (left_pos, RESIZE_HEIGHT * 1.0 / 2)
            right_point = (right_pos, RESIZE_HEIGHT * 1.0 / 2)
            points = [up_point, left_point, down_point, right_point]
            self.draw_points(real_img, points)
    #        print "up angle=" + str(up_angle)
    #        print "down angle=" + str(down_angle)

            up_line = self.make_hor_line(up_point, up_angle)
            down_line = self.make_hor_line(down_point, down_angle)
            left_line = self.make_ver_line(left_point, left_angle)
            right_line = self.make_ver_line(right_point, right_angle)
            """

            points = self.get_board_limits(up_line, right_line, down_line, left_line)
            if(DEBUG):
                self.draw_lines([up_line, left_line, right_line, down_line], edgeim)
                self.draw_points(real_img, points)
            proim = self.projection(points, real_img, [line_num,right_num,0 ,
                                                       left_num] )
            gui_img_manager.add_img(self.get_line_image(hor, edgeim))
            gui_img_manager.add_img(self.get_line_image(new_hor, edgeim))
            gui_img_manager.add_img(self.get_line_image([up_line, left_line, right_line, down_line], edgeim))
            gui_img_manager.add_img(proim)
            return proim
        except:
            print("cut board fixer has failed")
            return real_img

    def get_line_image(self, lines, img):
        bin = copy.deepcopy(img)
        bin = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)

        for i in range(0, len(lines)):
            l = lines[i]
            cv2.line(bin, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
        return bin


def test(foldername):
    # get lines from image, and edge-image
    id = identify_board.identify_board()
    fixer = board_cut_fixer()
    for j in range(300, 350):
        try:
            edgeim, realim = id.get_image_from_filename(foldername+"\\projected\\"+str(j)+".jpg",False)
            fixed_im = fixer.main(realim, edgeim)

            cv2.imwrite(foldername+"\\fixed\\"+str(j)+".jpg",fixed_im)
            print(j)
        except:
            print(str(j)+" failed")

#test('pictures')
