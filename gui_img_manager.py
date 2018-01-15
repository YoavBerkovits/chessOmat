import game_loop
from threading import Thread

### STATIC VARIABLES - GLOBAL - this is black sorcery so DON'T TOUCH it.
### You will NOT BE ABLE TO FIX THIS. NOBODY KNOWS HOW THIS WORKS. ###
images = []
move = ""
curr = 0
finished = False

def set_camera(num):
    global images
    if(num>=len(images)):
        images.append([]) #new camera
 
def reset_images(camnum):
    global images
    images[camnum] = []

def add_img(img):
    global images
    images[-1].append(img)

def set_move(mv):
    global move
    move = mv

def get_images():
    return images

def set_finished(state):
    global finished
    global images
    if state==False:
        images = []
    finished = state

def check_images():
    return finished

# run game loop processing ASYNC
def init(moves_file, img_dir_lst,with_saves):
    def asyn_run():
        loop = game_loop.game_loop(angles_num, real_moves,img_dir_lst, with_saves)
        loop.main()

    real_moves = []
    if(moves_file == None):
        real_moves = None
    else:
        for line in open(moves_file):
            move = line.rstrip('\n')
            real_moves.append((move[0:2], move[2:4]))

    if img_dir_lst == None:
        angles_num = 2
    else:
        angles_num = len(img_dir_lst)
    thrd = Thread(target=asyn_run)
    thrd.start()
