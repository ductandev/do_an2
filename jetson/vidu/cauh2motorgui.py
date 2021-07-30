import RPi.GPIO as GPIO
import tkinter as tk

GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(16, GPIO.OUT)

TK = tk.Tk()

def FOR_M1():
        GPIO.output(11, GPIO.HIGH)
        GPIO.output(12, GPIO.LOW)

def REV_M1():
        GPIO.output(11, GPIO.LOW)
        GPIO.output(12, GPIO.HIGH)
def STOP_M1():
        GPIO.output(11, GPIO.LOW)
        GPIO.output(12, GPIO.LOW)

Nut_FOR_M1 = tk.Button(TK, height = 5,width = 15, text ="FOR_M1", command = FOR_M1)
Nut_REV_M1 = tk.Button(TK, height = 5,width = 15, text ="REV_M1", command = REV_M1)
Nut_STOP_M1 = tk.Button(TK, height = 5,width = 15, text ="STOP_M1", command = STOP_M1)

Nut_FOR_M1.pack()
Nut_REV_M1.pack()
Nut_STOP_M1.pack()

def FOR_M2():
        GPIO.output(15, GPIO.HIGH)
        GPIO.output(16, GPIO.LOW)

def REV_M2():
        GPIO.output(15, GPIO.LOW)
        GPIO.output(16, GPIO.HIGH)
def STOP_M2():
        GPIO.output(15, GPIO.LOW)
        GPIO.output(16, GPIO.LOW)

Nut_FOR_M2 = tk.Button(TK, height = 5,width = 15, text ="FOR_M2", command = FOR_M2)
Nut_REV_M2 = tk.Button(TK, height = 5,width = 15, text ="REV_M2", command = REV_M2)
Nut_STOP_M2 = tk.Button(TK, height = 5,width = 15, text ="STOP_M2", command = STOP_M2)

Nut_FOR_M2.pack()
Nut_REV_M2.pack()
Nut_STOP_M2.pack()

TK.mainloop()
