import filter_colors
import identify_board
import board_cut_fixer
import cv2


print_and_save = True
class photos_angle:
    def __init__(self, hardware1, chess_helper1, self_idx):
        self.chess_helper = chess_helper1
        self.hardware = hardware1
        self.idx = self_idx
        self.boardid = identify_board.identify_board()
        self.fixer = board_cut_fixer.board_cut_fixer()
        
    def init_colors(self):
        cut_board_im = self.get_new_img()
        self.color_filter = filter_colors.filter_colors(cut_board_im, self.chess_helper)
     
    def prep_img(self):
        self.prep_im = self.hardware.get_image(self.idx)
    
    def get_new_img(self, dir_if_test = None):
        new_board_im = self.prep_im
        cut_board_im, edges = self.boardid.main(new_board_im)
        if print_and_save:
            if dir_if_test is not  None:
                cv2.imwrite(dir_if_test + 'first_cut_img.jpg', cut_board_im)
            else:
                cv2.imwrite('first_cut_img.jpg', cut_board_im)
        better_cut_board_im = self.fixer.main(cut_board_im, edges)
        if dir_if_test is not None:
            cv2.imwrite(dir_if_test + 'second_cut_img' + str(self.idx) + '.jpg', better_cut_board_im)
        else:
            cv2.imwrite('second_cut_img' + str(self.idx) + '.jpg', better_cut_board_im)
        return better_cut_board_im

    def get_square_diff(self, cut_board_im, src, is_source):
        return self.color_filter.get_square_diff(cut_board_im, src, is_source)

    def update_board(self, last_move):
        self.color_filter.update_board(last_move)

    def set_prev_im(self, img):
        return self.color_filter.set_prev_im(img)

