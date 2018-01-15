import os
import errno
import hardware as hw
import chess_helper as ch
import find_moves_rank as fm
import photos_angle
import chess_engine_wrapper
import gui_img_manager

"""
Main logic file.
"""
SOURCE = True
LEFT = 0
RIGHT = 1
ROWS_NUM = 8


class game_loop:

    def __init__(self, angles_num, real_moves_if_test = None, imgs_if_test = None, if_save_and_print = True):
        self.if_save_and_print = if_save_and_print
        self.moves_counter = -1
        self.black_im = self.create_black_im()
        if real_moves_if_test is not None:
            self.is_test = True
            self.real_moves = real_moves_if_test

        else:
            self.is_test = False

        self.hardware = hw.hardware(angles_num, imgs_if_test)
        self.my_turn = self.hardware.is_i_first()
        # TODO : get starting player type by color!

        self.chesshelper = ch.chess_helper(ch.chess_helper.ME)

        self.ph_angles = []
        gui_img_manager.set_finished(False)
        
        for i in range(angles_num):
            gui_img_manager.set_camera(i)
            self.ph_angles.append(photos_angle.photos_angle(self.hardware,self.chesshelper, i))
            self.ph_angles[i].prep_img()
            
        for ang in self.ph_angles:
            ang.init_colors()
            
        gui_img_manager.set_finished(True)

        self.movefinder = fm.find_moves_rank(self.chesshelper)

        self.chess_engine = chess_engine_wrapper.chess_engine_wrapper()
        self.last_move = None
        #TODO delete upper row

    def main(self):
        last_move = None
        while True:
            gui_img_manager.set_finished(False)
            if self.my_turn:
                best_move = self.chess_engine.get_best_move(last_move)
                print("I recommend: " + best_move)
                self.hardware.player_indication(best_move)
            last_move = self.get_new_move()
            self.chesshelper.do_turn(last_move[0], last_move[1])
            self.my_turn = not self.my_turn
            gui_img_manager.set_finished(True)

    def get_new_move(self):
        self.moves_counter += 1
        print("move num" + str(self.moves_counter))
        # for angle in self.ph_angles:
        #    angle.update_board(self.last_move)
        real_move = None
        if(self.is_test): 
            real_move = self.real_moves[self.moves_counter]
        relevant_squares = self.chesshelper.get_relevant_locations()
        sources = relevant_squares[0]
        dests = relevant_squares[1]
        pairs = []
        pairs_ranks = []
        for i in range(len(self.ph_angles)):
            gui_img_manager.set_camera(i)
            self.ph_angles[i].prep_img()
        
        for i in range(len(self.ph_angles)):
            while True:
                try:
                    gui_img_manager.set_camera(i)
                    pairs_and_ranks = self.check_one_direction(sources, dests, angle_idx=i)
                    break
                except:
                    print("id error plz take another photo k thnx")
                    gui_img_manager.reset_images(i)
                    self.ph_angles[i].prep_img()
                    
            pairs = pairs + pairs_and_ranks[0]
            pairs_ranks = pairs_ranks + pairs_and_ranks[1]
        best_pair_idx = [i for i in range(len(pairs_ranks)) if pairs_ranks[i] == max(pairs_ranks)][0]
        move = pairs[best_pair_idx]


        #if self.if_save_and_print:
        if True:
            # TODO change the fucking if
            print("detected_move")
            print(move)
            print('real_move')
            print(real_move)

        if self.is_test:
            move = real_move


        self.last_move = move
        self.chesshelper.do_turn(move[0], move[1])
        # TODO get it out of here
        return move

    def check_one_direction(self, sources, dests, angle_idx):
        make_dir( 'super tester results/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle_idx))
        angle_dir = 'super tester results/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle_idx) + '/'
        real_move = None
        angle = self.ph_angles[angle_idx]
        cut_board_im = angle.get_new_img(angle_dir)
        if self.if_save_and_print:
            print("angle_num_" + str(angle_idx))

            print("sources are:")
            print(sources)

            print("destinations are:")
            print(dests)

        else:
            real_move = None
            angle_dir = None
            
        if(self.is_test):
            real_move = self.real_moves[self.moves_counter]
            
        sourcesims, sourcesabvims = self.get_diff_im_and_dif_abv_im_list(sources, cut_board_im, angle,
                                                                               SOURCE)
        destsims, destsabvims = self.get_diff_im_and_dif_abv_im_list(dests, cut_board_im, angle,
                                                                          not SOURCE)

        pairs, pairs_rank = self.movefinder.get_move(sources, sourcesims, sourcesabvims,
                                                     dests, destsims, destsabvims, real_move, angle_dir)

        ### save prev picture ###
        angle.set_prev_im(cut_board_im)

        return pairs, pairs_rank


    def get_diff_im_and_dif_abv_im_list(self, locs, cut_board_im, angle, is_source):
        angle_dir = 'super tester results/move_num_' + str(self.moves_counter) + '/angle_num_' + str(angle.idx) + '/'
        
        locssims = []
        locsabvims = []
        for loc in locs:
            abv_loc = self.get_abv_loc(loc)
            bel_loc = self.get_bel_loc(loc)
            diff_im = angle.get_square_diff(cut_board_im, loc, is_source)
            if abv_loc:
                diff_abv_im = angle.get_square_diff(cut_board_im, abv_loc, is_source)
            else:
                diff_abv_im = self.black_im
            #if self.if_save_and_print:
             #   if loc == real_move[0] or loc == real_move[1] or bel_loc == real_move[0] or bel_loc == real_move[1]:
              #      cv2.imwrite(angle_dir + loc + '.jpg', diff_im)
            locssims.append(diff_im)
            locsabvims.append(diff_abv_im)
        return locssims, locsabvims

    def get_abv_loc(self, loc):
        column = loc[0]
        row_num = int(loc[1])
        if row_num < 8:
            return column+str(row_num + 1)
        else:
            return False

    def get_bel_loc(self, loc):
        column = loc[0]
        row_num = int(loc[1])
        return column+str(row_num -1)

    def create_black_im(self):
        black_im = []
        for i in range(20):
            black_im.append([])
            for j in range(20):
                black_im[i].append(0)
        return black_im

def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise




