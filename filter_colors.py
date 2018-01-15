import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper
import chess
import cv2

BLACK = (0.0, 0.0, 0.0)
MINIMAL_PLAYER_BOARD_RATIO = 0.2
PIXELS_FOR_MAIN_COLORS = (200, 200)
PIXELS_SQUARE = (20, 20)
BLACK_NUM = 1
WHITE_NUM = 2
UP = True


class filter_colors:
    """
    This file is responsible for filtering and cataloguing the colors in the
    picture.
    """

    def __init__(self, im, chess_helper):
        self.chess_helper = chess_helper
        self.initialize_colors(im,chess_helper.user_starts)
        self.initialize_board()

    def initialize_board(self):
        self.board = []
        for i in range(8):
            self.board.append([])
            for j in range(8):
                self.board[i].append(None)

    def update_board(self, last_move):
        if last_move != None:
            row1 = ord(last_move[0]) - ord('a')
            colon1 = int(last_move[1]) - 1
            row2 = ord(last_move[2]) - ord('a')
            colon2 = int(last_move[3]) - 1
            if colon1 < 6:
                self.board[row1][colon1 + 2] = None
            if colon1 < 6:
                self.board[row2][colon2 + 2] = None

    def color_dist(self, color1, color2):
        return abs(max(color1) - max(color2) - min(color2) + min(color1))

    def cmpT(self, t1, t2):
        return t1[0] == t2[0] and t1[1] == t2[1] and t1[2] == t2[2]

    """
      gets all 4/3/2 colors in the board.
      :im initial image, that contains the relevant colors on the board.
      :return nothing
      """

    def initialize_colors(self, im,is_me_white):
        self.main_colors = self.get_main_colors(im,is_me_white)
        self.set_prev_im(im)

    """
        :return black,white,my soldier color, rival soldier color.
        """

    def get_main_colors(self, im,is_me_white):
        im_resize = cv2.resize(im, PIXELS_FOR_MAIN_COLORS)
        board_colors = self.get_board_colors(im_resize)
        down_color = self.get_player_color(im_resize, board_colors, not UP,is_me_white)
        up_color = self.get_player_color(im_resize, board_colors, UP,is_me_white)
        main_colors = board_colors
        main_colors.append(down_color)
        main_colors.append(up_color)
        self.set_colors_nums_and_relevant_changes(main_colors)
        print('main colors are:')
        print(main_colors)
        return main_colors

    """
        gets 2 primary colors from board image.
        """

    def get_board_colors(self, im):
        # TODO fix this lines
        im_sz = len(im)
        ar = im[(im_sz // 4):(3 * im_sz // 4)]
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, 2)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
        indices = [i[0] for i in
                   sorted(enumerate(-counts), key=lambda x: x[1])]
        new_indices = []
        new_codes = []
        for i in indices:
            new_codes.append(codes[i])
        if self.color_dist(new_codes[0], BLACK) < self.color_dist(new_codes[1],
                                                                  BLACK):
            new_indices.append(indices[0])
            new_indices.append(indices[1])
        else:
            new_indices.append(indices[1])
            new_indices.append(indices[0])
        return [codes[i] for i in new_indices]

    """
        gets player's colors from the board image.
        """

    def get_player_color(self, im, board_colors, is_up,is_me_white):
        # ar = np.asarray(im)
        ar = im
        # TODO fix these lines
        ar_sz = len(ar)
        if is_up:
            ar = ar[:(ar_sz // 4)]
        else:
            ar = ar[3 * (ar_sz // 4):]
        shape = ar.shape
        ar2 = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar2, 3)
        for i in range(3):
            color_dist_1 = self.color_dist(codes[i], board_colors[0])
            color_dist_2 = self.color_dist(codes[i], board_colors[1])
            if i == 0:
                max_dist = min(color_dist_1, color_dist_2)
                max_dist_index = i
            else:
                dist_i = min(color_dist_1, color_dist_2)
                if dist_i > max_dist:
                    max_dist = dist_i
                    max_dist_index = i
        player_color = codes[max_dist_index]
        num_of_player_pix = 0
        for rowidx in range(len(ar)):
            row = ar[rowidx]
            for pix in row:
                color_dist_from_player = self.color_dist(pix, player_color)
                color_dist_from_black = self.color_dist(pix, board_colors[0])
                color_dist_from_white = self.color_dist(pix, board_colors[1])
                if color_dist_from_player < color_dist_from_black and \
                        color_dist_from_player < color_dist_from_white:
                    num_of_player_pix += 1
        num_of_pix = len(ar) * len(ar[0])
        rank = num_of_player_pix / num_of_pix
        print(rank)
        if rank < MINIMAL_PLAYER_BOARD_RATIO:
            if (is_me_white and is_up) or (not is_me_white and not is_up):
                player_color = board_colors[0]
            else:
                player_color = board_colors[1]
        return player_color

    def set_colors_nums_and_relevant_changes(self, main_colors):
        self.ME_NUM = 4
        self.HIM_NUM = 8

        if self.cmpT(main_colors[2], main_colors[0]):
            self.ME_NUM = BLACK_NUM
        elif self.cmpT(main_colors[2], main_colors[1]):
            self.ME_NUM = WHITE_NUM
        if self.cmpT(main_colors[3], main_colors[0]):
            self.HIM_NUM = BLACK_NUM
        elif self.cmpT(main_colors[3], main_colors[1]):
            self.HIM_NUM = WHITE_NUM

        self.RELEVANT_CHANGES_ME_BLACK_SOURCE = [BLACK_NUM - self.ME_NUM, self.HIM_NUM - self.ME_NUM]
        self.RELEVANT_CHANGES_ME_WHITE_SOURCE = [WHITE_NUM - self.ME_NUM, self.HIM_NUM - self.ME_NUM]
        self.RELEVANT_CHANGES_HIM_BLACK_SOURCE = [BLACK_NUM - self.HIM_NUM, self.ME_NUM - self.HIM_NUM]
        self.RELEVANT_CHANGES_HIM_WHITE_SOURCE = [WHITE_NUM - self.HIM_NUM, self.ME_NUM - self.HIM_NUM]
        self.RELEVANT_CHANGES_ME_BLACK_TARGET = [self.ME_NUM - BLACK_NUM, self.ME_NUM - self.HIM_NUM]
        self.RELEVANT_CHANGES_ME_WHITE_TARGET = [self.ME_NUM - WHITE_NUM, self.ME_NUM - self.HIM_NUM]
        self.RELEVANT_CHANGES_HIM_BLACK_TARGET = [self.HIM_NUM - BLACK_NUM, self.HIM_NUM - self.ME_NUM]
        self.RELEVANT_CHANGES_HIM_WHITE_TARGET = [self.HIM_NUM - WHITE_NUM, self.HIM_NUM - self.ME_NUM]

        if self.ME_NUM == BLACK_NUM :
            self.RELEVANT_CHANGES_ME_BLACK_SOURCE = [self.HIM_NUM - self.ME_NUM]
            self.RELEVANT_CHANGES_ME_BLACK_TARGET = [self.ME_NUM - self.HIM_NUM]
        elif self.ME_NUM == WHITE_NUM:
            self.RELEVANT_CHANGES_ME_WHITE_SOURCE = [self.HIM_NUM - self.ME_NUM]
            self.RELEVANT_CHANGES_ME_WHITE_TARGET = [self.ME_NUM - self.HIM_NUM]
        if self.HIM_NUM == BLACK_NUM:
            self.RELEVANT_CHANGES_HIM_BLACK_SOURCE = [self.ME_NUM - self.HIM_NUM]
            self.RELEVANT_CHANGES_HIM_BLACK_TARGET = [self.HIM_NUM - self.ME_NUM]
        elif self.HIM_NUM == WHITE_NUM:
            self.RELEVANT_CHANGES_HIM_WHITE_SOURCE = [self.ME_NUM - self.HIM_NUM]
            self.RELEVANT_CHANGES_HIM_WHITE_TARGET = [self.HIM_NUM - self.ME_NUM]
        return

    ###########################################################################

    """
        sets previous image - in rgb format.
        """

    def set_prev_im(self, im):
        self.prev_im = im

    """
    :im image of square after turn NOT CATALOGUED
    :square_loc location of square, in uci format
    :return binary image of relevant differences only (according to
    player/square color)
    """

    def get_square_diff(self, im, square_loc, is_source):
        is_white = self.chess_helper.square_color(square_loc) == chess.WHITE
        row = ord(square_loc[0]) - ord('a')
        colon = int(square_loc[1]) - 1
        after_square = cv2.resize(self.get_square_image(im, square_loc, self.chess_helper.user_starts), PIXELS_SQUARE)
        after_square = self.catalogue_colors(after_square, is_white)
        #maybe_before_square = self.board[row][colon]
        maybe_before_square = None
        #TODO fix it (now we are not using the saved board
        if maybe_before_square == None:
            before_square = cv2.resize(self.get_square_image(self.prev_im, square_loc, self.chess_helper.user_starts), PIXELS_SQUARE)
            before_square = self.catalogue_colors(before_square, is_white)
        else:
            before_square = maybe_before_square
        self.board[row][colon] = after_square
        square_diff = self.make_binary_relevant_diff_im(before_square, after_square, is_white, is_source,self.chess_helper.curr_player)
        square_diff_2 = np.array(square_diff)
        # square_diff_cut = cv2.cvtColor(square_diff_2, cv2.COLOR_GRAY2RGB)
        # print(square_diff_cut)
        return square_diff

    """
    :return an image with 4/3/2 colors only.
    """

    """
    returns subimage of a square in the board.
    """

    def get_square_image(self, im, loc, did_I_start):
        locidx = self.chess_helper.ucitoidx(loc)
        sq_sz = len(im[0]) // 8
        sq_sz_y = len(im)//8
        x = locidx[0]
        if did_I_start:
            y = 8 - locidx[1]
        else:
            y = locidx[1]
        area = (x * sq_sz, y * sq_sz_y, (x + 1) * sq_sz, (y + 1) * sq_sz_y)
        sqr_im = im[area[1]:area[3], area[0]:area[2]]
        return sqr_im

    def catalogue_colors(self, im, is_white):
        main_colors_white = self.main_colors[1:]
        main_colors_black = [self.main_colors[0]] + self.main_colors[2:]
        if is_white:
            cat_im = self.fit_colors(im, main_colors_white, is_white)
        else:
            cat_im = self.fit_colors(im, main_colors_black, is_white)
        return cat_im

    """
    :return image fit to 4 main colors.
    """

    def fit_colors(self, im, main_colors, is_white):
        new_im = []
        for rowidx in range(len(im)):
            i = 0
            row = im[rowidx]
            new_im.append([])
            for pix in row:
                min_dist = self.color_dist(pix, main_colors[0])
                if is_white:
                    new_im[rowidx].append(WHITE_NUM)
                    if not self.cmpT(main_colors[1], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[1])
                        if dist < min_dist:
                            min_dist = dist
                            new_im[rowidx][i] = self.ME_NUM
                    if not self.cmpT(main_colors[2], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[2])
                        if dist < min_dist:
                            new_im[rowidx][i] = self.HIM_NUM
                else:
                    new_im[rowidx].append(BLACK_NUM)
                    if not self.cmpT(main_colors[1], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[1])
                        if dist < min_dist:
                            min_dist = dist
                            new_im[rowidx][i] = self.ME_NUM
                    if not self.cmpT(main_colors[2], main_colors[0]):
                        dist = self.color_dist(pix, main_colors[2])
                        if dist < min_dist:
                            new_im[rowidx][i] = self.HIM_NUM
                i += 1
        return new_im

    def make_binary_relevant_diff_im(self, im1, im2, is_white, is_source,my_turn):
        if my_turn:
            if is_white:
                if is_source:
                    relevant_changes = self.RELEVANT_CHANGES_ME_WHITE_SOURCE
                else:
                    relevant_changes = self.RELEVANT_CHANGES_ME_WHITE_TARGET
            else:
                if is_source:
                    relevant_changes = self.RELEVANT_CHANGES_ME_BLACK_SOURCE
                else:
                    relevant_changes = self.RELEVANT_CHANGES_ME_BLACK_TARGET
        else:
            if is_white:
                if is_source:
                    relevant_changes = self.RELEVANT_CHANGES_HIM_WHITE_SOURCE
                else:
                    relevant_changes = self.RELEVANT_CHANGES_HIM_WHITE_TARGET
            else:
                if is_source:
                    relevant_changes = self.RELEVANT_CHANGES_HIM_BLACK_SOURCE
                else:
                    relevant_changes = self.RELEVANT_CHANGES_HIM_BLACK_TARGET
        binary_im = []
        for rowidx in range(len(im1)):
            binary_im.append([])
            for pixidx in range(len(im1[0])):
                if im2[rowidx][pixidx] - im1[rowidx][pixidx] in relevant_changes:
                    binary_im[rowidx].append(1)
                else:
                    binary_im[rowidx].append(0)
        return binary_im
