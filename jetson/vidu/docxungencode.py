import RPi.GPIO as GPIO
import time

# Pin Definitons:
pin_out = 12  # BOARD pin 12
pin_encode = 18  # BOARD pin 18

def main():
    prev_value = 100

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BOARD pin-numbering scheme
    GPIO.setup(pin_out, GPIO.OUT)  # LED pin set as output
    GPIO.setup(pin_encode, GPIO.IN)  # Button pin set as input

    # Initial state for LEDs:
    GPIO.output(pin_out, GPIO.HIGH)
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            curr_value = GPIO.input(pin_encode)
            if curr_value <= prev_value:
                print("Outputting {} to Pin {}".format(curr_value, pin_encode))
    finally:
        GPIO.cleanup()  # cleanup all GPIO

if __name__ == '__main__':
    main()