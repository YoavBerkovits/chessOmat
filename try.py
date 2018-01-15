import RPi.GPIO as GPIO
import time
import picamera


   
def one_still(a):
    print(a)
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        camera.start_preview()
        # Camera warm-up time
        time.sleep(0.2)
        camera.capture(str(a)+'.jpg')
        return(a+1)


GPIO.setmode(GPIO.BOARD)
GPIO.setup(7,GPIO.IN,pull_up_down = GPIO.PUD_DOWN)
should_enter = True
counter = 0
try:
    while True:
        if(GPIO.input(7) == 1):
            should_enter = True
            
        elif(should_enter):
            one_still(counter)
            counter = counter+1
            should_enter = False

except KeyboardInterrupt:
    GPIO.cleanup()
    
    
 