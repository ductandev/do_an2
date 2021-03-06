import RPi.GPIO as GPIO
import time
LedPin = 11                         # pin11

GPIO.setmode(GPIO.BOARD)            # Numbers GPIOs by physical location
GPIO.setup(LedPin, GPIO.OUT)        # Set LedPin's mode is output
GPIO.output(LedPin, GPIO.HIGH)      # Set LedPin high(+3.3V) to turn on led

try:
  while True:
    GPIO.output(LedPin, GPIO.HIGH)  # led on
    time.sleep(1)
    GPIO.output(LedPin, GPIO.LOW)   # led off
    time.sleep(1)
except KeyboardInterrupt:           # When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
  pass
  GPIO.output(LedPin, GPIO.LOW)     # led off
  GPIO.cleanup()                    # Release resource