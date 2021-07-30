import Jetson.GPIO as GPIO
import time
led_pin = 19
butt = 37

GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(butt, GPIO.IN)
GPIO.setwarnings(False)
#GPIO.setup(38, GPIO.OUT)
cnt = 0
while True:
    if(GPIO.input(butt) == GPIO.LOW):
        print("taolaobidao")
        if(cnt == 0):
            cnt = 1
            GPIO.output(led_pin, GPIO.HIGH)
            time.sleep(1)
        else:
            cnt = 0
            GPIO.output(led_pin, GPIO.LOW)
            time.sleep(1)