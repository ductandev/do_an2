import RPi.GPIO as GPIO
import time

# Pin Definitions
output_pout = 12  # BCM pin 18, BOARD pin 12
input_pin = 11
def main():
    prev_value = None

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    GPIO.setup(input_pin, GPIO.IN) # set pin as an input pin
    GPIO.setup(output_pout, GPIO.OUT, initial=GPIO.HIGH) # set pin as an input out
    print("Starting demo now! Press CTRL+C to exit")
    try:
        while True:
            value = GPIO.input(input_pin)
            if value != prev_value:
                if value == GPIO.HIGH:
                    value_str = "HIGH"
                else:
                    value_str = "LOW"
                print("Value read from pin {} : {}".format(input_pin,
                                                           value_str))
                prev_value = value
            time.sleep(1)
    finally:
        GPIO.cleanup()

if __name__ == '__main__':
    main()
