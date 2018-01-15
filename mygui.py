from tkinter import *
import chess
import scipy

from scipy import misc
from PIL import Image
from PIL import ImageTk
import listener
from threading import Thread
import time
from multiprocessing import Process
import cv2
import gui_img_manager
import copy

###dictionary
DICTIONARY = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}

class GameBoard():
    def __init__(self,root,canvas, rows=8, columns=8, size=640,
                 color1="white",
                 color2="#AAAAAF"):
        '''size is the size of a square, in pixels'''
        self.root = root
        self.canvas = canvas
        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.x = self.size*7/7
        self.y = self.size*6/7
        self.pieces = {}
        self.position_of_board = [["r","n","b","q","k","b","n","r"],
                                  ["p", "p", "p", "p", "p", "p", "p","p"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["O","O","O","O","O","O","O","O"],
                                  ["wp","wp","wp","wp","wp","wp","wp","wp"],
                                  ["wr", "wn", "wb", "wq", "wk", "wb", "wn",
                                   "wr"]]

        self.canvas_width = columns * size
        self.canvas_height = rows * size
        self.board_img = self.make_img_from_file("board.png",self.size,self.size)
        self.white_players_turn = self.make_img_from_file(
            "white_players_turn.png",70,300)
        self.black_players_turn = self.make_img_from_file(
            "black_players_turn.png",70,300)
        self.board_state = self.make_img_from_file(
            "board_state.png",50,250)
        self.line_analysis = self.make_img_from_file(
            "line_analysis.png",50,250)
        self.turn = 1

    def make_img_from_file(self,file_name,x,y):
        img = Image.open(file_name)
        resize_img = Image.fromarray(misc.imresize(img, (x,y)))
        final_img = ImageTk.PhotoImage(resize_img)
        return final_img

    def chaneg_player(self):
        self.turn = self.turn + 1
        if self.turn%2 == 0:
            self.canvas.create_image(600, 40,image =self.white_players_turn)
        elif self.turn%2 == 1:
            self.canvas.create_image(600, 40, image =self.black_players_turn)
        return self.white_players_turn,self.black_players_turn

    def draw_board(self):
        '''draw board'''
        self.canvas.create_image(self.x, self.y, image=self.board_img)
        self.canvas.create_image(self.x, self.y-self.size/2-35,
                                 image=self.board_state)
        #self.canvas.create_image(self.x+100,self.y+100, image = self.r_img)
        return self.board_img #, self.r_img

    def placepiece(self, imageName, row, column):
        '''Place a piece at the given row/column'''
        image = Image.open(imageName)
        resize_img = Image.fromarray(misc.imresize(image, (self.size//8
                                                           ,self.size//8)))
        image = ImageTk.PhotoImage(resize_img)
        x0 = int(self.x+((column-8) * self.size//8)) + int(self.size//2)+ \
             int(self.size//16)
        y0 = int(self.y+((row-8) * self.size//8)) + int(self.size//2)+int(self.size//16)
        self.canvas.create_image( x0, y0,image = image)
        return image

    def draw_position_of_board(self):

        '''draw board and pieces'''
        images = []
        board_img = self.draw_board()
        for i in range(len(self.position_of_board)):
            for j in range(len(self.position_of_board[0])):
                if self.position_of_board[i][j] !="O":
                    images.append(self.placepiece(self.position_of_board[i][
                                              j]+".png",i,j))
        self.chaneg_player()
        return images, board_img

    def make_move(self,move):
        row = DICTIONARY[move[0][0]]
        colomn = int(move[0][1])+1
        new_row = DICTIONARY[move[1][0]]
        new_colomn = int(move[1][1])+1

        moved_piece = self.position_of_board[row][colomn]

        str_move = move[0]+move[1]
        if len(str_move)==5:        #promotion
            if moved_piece[0]=="w":
                self.position_of_board[new_row][new_colomn] = "w"+str_move[4]
            else:
                self.position_of_board[new_row][new_colomn] = str_move[4]
                self.position_of_board[row][colomn] = "O"
        else:
            if str_move == "e1g1" and moved_piece == "wk":      #castling
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("h1","f1"))
            elif str_move == "e1b1" and moved_piece == "wk":
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("a1","c1"))

            elif str_move == "e8g8" and moved_piece == "k":
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("h8","f8"))
            elif str_move == "e8b8" and moved_piece == "k":
                self.position_of_board[new_row][new_colomn] = moved_piece
                moved_piece(("a8","c8"))
            else:       #regular move

                self.position_of_board[new_row][new_colomn] = moved_piece
                self.position_of_board[row][colomn] = "O"



        images = self.draw_position_of_board()
        return images

    def refresh(self, event):
        '''Redraw the board, possibly in response to window being resized'''
        xsize = int((event.width-1) / self.columns)
        ysize = int((event.height-1) / self.rows)
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2
        for name in self.pieces:
            self.placepiece(name, self.pieces[name][0], self.pieces[name][1])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

class GIF():
    def __init__(self, x, y,size, files_names_lst,root):
        self.root = root
        self.x = x
        self.y = y
        self.files_names_lst = files_names_lst
        self.images = []
        self.gif_counter = 0
        self.size =size


    def draw_gif(self):
        for i in range(len(self.files_names_lst)):
            image = Image.open(self.files_names_lst[i])
            resize_img = Image.fromarray(misc.imresize(image,
                                                       (self.size,self.size)))
            final_image = ImageTk.PhotoImage(resize_img)
            self.images.append(final_image)
        timer_label = Label(self.root, text="")
        timer_label.place(anchor=NW, x=self.x, y=self.y)
        timer_label.configure(image=self.images[self.gif_counter])
        self.gif_counter = (self.gif_counter + 1) % len(self.images)
        x = self.root.after(3000, self.draw_gif)

    def draw_gif2(self):
        timer_label = Label(self.root, text="")
        timer_label.place(anchor=NW, x=self.x, y=self.y)
        timer_label.configure(image=self.images[self.gif_counter])
        self.gif_counter = (self.gif_counter + 1) % len(self.images)
        x = self.root.after(3000, self.draw_gif)

class CLOCK():
    def __init__(self,root,canvas):
        self.root = root
        self.now = 0
        self.images = []
        self.canvas = canvas
        self.Hight = 70
        self.wight = 45
        for i in range(10):
            num = Image.open(str(i)+"1.png")
            resize_img = Image.fromarray(misc.imresize(num,
                                                       ( self.Hight,self.wight)))
            final_image = ImageTk.PhotoImage(resize_img)
            self.images.append(final_image)

        oo = Image.open("001.png")
        resize_oo = Image.fromarray(misc.imresize(oo,
                                                  (self.Hight,
                                    int(self.Hight/5+1))))
        self.oo = ImageTk.PhotoImage(resize_oo)

    def update_clock(self):
        for img in self.images:
            self.canvas.delete(img)
        self.canvas.create_image(1000-self.wight, self.Hight/2,
                                 image=self.images[
            int(self.now%10)])
        self.canvas.create_image(1000-2*self.wight, self.Hight/2,
                                 image=self.images[
                                     int(self.now/10)%6])
        self.canvas.create_image(1000-3*self.wight+int(self.Hight/5)+2,
                                 self.Hight/2,image =
        self.oo)
        self.canvas.create_image(1000-3*self.wight-int(self.Hight/5),
                                 self.Hight/2,
                                 image=self.images[
            int(self.now/60)%10])
        self.canvas.create_image(1000-4*self.wight-int(self.Hight/5),
                                 self.Hight/2, image=self.images[
                                     int(self.now/600)%10])


        self.now = self.now + 1
        x = self.root.after(990, self.update_clock)

class GUI():

    def __init__(self,gif_images_lsts):
        self.WINDOW_WIDTH = 1000
        self.WINDOW_HEIGHT = 600
        self.root = Tk()
        self.now = 0
        self.root.geometry("1000x1000")
        self.canvas = Canvas(self.root, width=self.WINDOW_WIDTH,
                        height=self.WINDOW_HEIGHT,bg = "black")
        self.canvas.pack()
        self.gif_images_lsts=gif_images_lsts
        self.gif_counter = 0

        self.x = 600
        self.y = 200

        self.gif_x = 0
        self.gif_y = 0

        self.garbage = []
        self.board = GameBoard(self.root, self.canvas,size =
        int(self.WINDOW_WIDTH*0.5*(1-1/len(self.gif_images_lsts))))

        self.images_for_realtime_gifs = [[],[]]
        self.runtime_counter = 0

        self.thread = None

    def old_update_clock(self):
        timer_label = Label(text="")
        timer_label = Label(self.root, text="", bg="#AFFAAA",
                            fg="orange",justify=RIGHT)
        timer_label.place(anchor=NW, x=self.WINDOW_WIDTH-self.x, y=0)
        timer_label.configure(text=str(self.now // 60) + ':' + str(self.now %
                                                                   60),
                              highlightbackground  = "red",
                              font=("Courier", 44))
        self.now = self.now + 1
        x = self.root.after(1000, self.update_clock)

    def draw_gifs(self):
        gifs = []
        for i in range(len(self.gif_images_lsts)):
            gifs.append( GIF(0,i*self.WINDOW_HEIGHT/len(self.gif_images_lsts),
                          int(self.WINDOW_HEIGHT/len(self.gif_images_lsts)),
                          self.gif_images_lsts[i],self.root))
        for gif in gifs:
            gif.draw_gif()

        return gifs

    def draw_realtime_gifs(self):
        real_time_gifs = []
        for i in range(2):
            real_time_gifs.append( GIF(self.WINDOW_WIDTH*7/10,
                                       (8*i+3)*self.WINDOW_HEIGHT*1/18,
                          int(self.WINDOW_HEIGHT*7/18),
                                       self.images_for_realtime_gifs[i],self.root))
        for gif in real_time_gifs:
            gif.draw_gif2()

        self.garbage.append( real_time_gifs)
        while True:
            time.sleep(1)
            if gui_img_manager.check_images():
                gui.images_for_realtime_gifs = gui_img_manager.get_images()
                gui.runtime_counter = 0

    def draw_clock(self):
        clock = CLOCK(self.root,self.canvas)
        clock.update_clock()
        return clock

    def draw_image_from_file(self, filename,x,y):
        image = Image.open(filename)
        resize_img = Image.fromarray(misc.imresize(image, (230,
                                                           230)))
        final_image = ImageTk.PhotoImage(resize_img)
        self.canvas.create_image(x, y, image=final_image)
        self.garbage.append( final_image)

    def draw_image(self, img,x,y):


        resize_img = Image.fromarray(misc.imresize(img, (230,230)))
        imgcopy = resize_img.copy()
        final_image = ImageTk.PhotoImage(imgcopy)
        self.canvas.create_image(x, y, image=final_image)
        self.garbage.append(imgcopy)
        self.garbage.append(final_image)

        return final_image

    def draw_board(self):
        images = self.board.draw_board()
        images2 = self.board.draw_position_of_board()
        return images, images2

    def make_move(self,move):
        images = self.board.make_move(move)
        self.garbage.append(images)
        return self.board

    def not_got_image(self):
        return False

    def getImage(self):
        return None

    def changeImage(self, im):
        return False

    def shimri(self):
        while True:
            print("shimri")
            time.sleep(1)

    def server_wait_image(self):
        '''''
        while True:
            while (self.not_got_image()): pass
            im = self.getImage()
            self.changeImage(im)
        '''''
        return  None

    def inon(self):
        while True:
            print ("inon")
            time.sleep(1)

    def set_images_for_real_time(self, files_names_lst):
        for i in range(2):
            for file_name in files_names_lst[i]:
                img = Image.open(file_name)
                final_img = ImageTk.PhotoImage(img)
                self.images_for_realtime_gifs[i].append(final_img)

    def draw_next_runtime(self):

        for i in range(2):
            x = self.WINDOW_WIDTH * 7 / 10+int(self.WINDOW_HEIGHT * 7 / 36)
            y = (8 * i + 3) * self.WINDOW_HEIGHT * 1 / 18+int(self.WINDOW_HEIGHT * 7 / 36)

            num_imgs = len(self.images_for_realtime_gifs[i])
            img = self.images_for_realtime_gifs[i][
                                self.runtime_counter % num_imgs]


            self.draw_image(img,x,y)
        self.runtime_counter = self.runtime_counter+1


    def make_next_button(self):
        b = Button(self.root, text="get \n next \n image", command =
        self.draw_next_runtime)
        b.place(x=950, y=self.WINDOW_HEIGHT/5)

    def make_auto_button(self,):
        b = Button(self.root, text="auto \n run",command =
        self.draw_realtime_gifs)
        b.place(x=950, y=self.WINDOW_HEIGHT / 5+80)



if __name__ == "__main__":
    gui = GUI([["our_process.png"],["one.png", "two.png", "three.png",
                                 "four.png"],
               [ "Screenshot_2.png",
                "Screenshot_4.png",
                "Screenshot_5.png",
                "Screenshot_7.png"],  ["one1.png",
                                                                 "three1.png",
                                                                 "four1.png",
                                                                 "five1.png"],
               ["Small-mario.png",
                "0.png", "1.png"]])
    gui_img_manager.init(None, None,True)
    gui.canvas.pack()
    a = gui.draw_gifs()
    clock = gui.draw_clock()
    d = gui.draw_board()
    c = gui.make_move
    gui.root.after(1000, gui.make_move, ("b3", "e4"))
#    gui.images_for_realtime_gifs=[["one.png", "two.png", "three.png",
#                                     "four.png"],
#              [ "Screenshot_2.png","Screenshot_4.png", "Screenshot_5.png","Screenshot_7.png"]]

#    gui.images_for_realtime_gifs = get_images()

    gui.make_next_button()
    gui.make_auto_button()

    while True:
        if gui_img_manager.check_images():
            images = gui_img_manager.get_images()
            gui.images_for_realtime_gifs = images
            print(images)
            break
    gui.root.mainloop()

'''
    time.sleep(10)
    while True:
        time.sleep(1)
        if check_images():
            gui.images_for_realtime_gifs = get_images()
            gui.runtime_counter = 0



    while True:
        img1  = open("cat.png")
        img2 = open("r.png")
        gui.thread = Thread(target=gui.draw_image,args = (img1, 800,300))
        gui.thread.start()
        time.sleep(5)
        gui.thread = Thread(target=gui.draw_image,args = (img2, 800,300))
        gui.thread.start()


#get_images()
#check_images()
#    lsnr = listener.listener()

#    img1 = lsnr.get_image()
#    img2 = lsnr.get_image
#    cv2.imshow("1",img1)
#    cv2.waitKey(0)

#    gui.draw_image(img1,700,700)

    #t1 = threading.Thread(target=gui.server_wait_image())
#    p1 = Process(target=gui.inon())
#    p1.start()
#    p2 = Process(target=gui.server_wait_image())
#    p2.start()
#    p1.join()
#    p2.join()


    t2 = threading.Thread(target=gui.inon())
    t2.start()
    #t1.start()

    gui.root.after(1000, gui.make_move, ("b3", "e4"))
        '''



'''

img = scipy.misc.imread("0.png", flatten=False, mode="RGBA")
for i in len(img):
    for j in len (img[0]):
        img[i][j][3] = 0.5
scipy.cv2.imwrite("0", img, format="png")
'''
