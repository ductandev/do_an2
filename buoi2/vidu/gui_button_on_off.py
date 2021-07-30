import RPi.GPIO as GPIO
import tkinter

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)

TK = tkinter.Tk()


def BAT():
        GPIO.output(11, GPIO.HIGH)


def TAT():
        GPIO.output(11, GPIO.LOW)

Nut_Bat = tkinter.Button(TK, height = 5, width = 15, text ="ON", command = BAT)
Nut_Tat = tkinter.Button(TK, height = 5,width = 15, text ="OFF", command = TAT)


Nut_Bat.pack()
Nut_Tat.pack()

TK.mainloop()
