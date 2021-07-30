import Jetson.GPIO as GPIO
import time
led_pin = 19
led2_pin = 37
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(led2_pin, GPIO.IN)
while True:
    #GPIO.output(led2_pin, GPIO.HIGH)
    GPIO.output(led_pin,GPIO.HIGH)
    time.sleep(2)
   # GPIO.output(led2_pin,GPIO.LOW)
    GPIO.output(led_pin,GPIO.LOW)
    time.sleep(2)
