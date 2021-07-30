import RPi.GPIO as GPIO
import tkinter as tk

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)


TK = tk.Tk()


def BAT():
        GPIO.output(18, GPIO.LOW)

def TAT():
        GPIO.output(18, GPIO.HIGH)

Nut_Bat = tk.Button(TK, height = 5,width = 15, text ="ON", command = BAT)
Nut_Tat = tk.Button(TK, height = 5,width = 15, text ="OFF", command = TAT)

Nut_Bat.pack()
Nut_Tat.pack()

TK.mainloop()