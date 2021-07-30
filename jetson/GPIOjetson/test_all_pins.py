from __future__ import print_function
import sys
import time

import RPi.GPIO as GPIO

pin_datas = {
    'JETSON_NANO': {
        'unimplemented': (),
        'input_only': (),
    },
}
pin_data = pin_datas.get(GPIO.model)
all_pins = (7, 11, 12, 13, 15, 16, 18, 19, 21, 22, 23, 24, 26, 29, 31, 32, 33,
            35, 36, 37, 38, 40,)

if len(sys.argv) > 1:
    all_pins = map(int, sys.argv[1:])

for pin in all_pins:
    if pin in pin_data['unimplemented']:
        print("Pin %d unimplemented; skipping" % pin)
        continue

    if pin in pin_data['input_only']:
        print("Pin %d input-only; skipping" % pin)
        continue

    print("Testing pin %d as OUTPUT; CTRL-C to test next pin" % pin)
    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        while True:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.25)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
