import RPi.GPIO as GPIO
import Tkinter as tk

GPIO.setmode(GPIO.BCM)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)

GPIO.setup(12, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)

TK = tk.Tk()

def F_M1():
        GPIO.output(7, GPIO.HIGH)
	GPIO.output(11, GPIO.LOW)

def R_M1():
        GPIO.output(7, GPIO.LOW)
	GPIO.output(11, GPIO.HIGH)

def S_M1():
        GPIO.output(7, GPIO.LOW)
	GPIO.output(11, GPIO.LOW)

Nut_F_M1 = tk.Button(TK, height = 5,width = 15, text ="F_M1", command = F_M1)
Nut_R_M1 = tk.Button(TK, height = 5,width = 15, text ="R_M1", command = R_M1)
Nut_S_M1 = tk.Button(TK, height = 5,width = 15, text ="S_M1", command = S_M1)

Nut_F_M1.pack()
Nut_R_M1.pack()
Nut_S_M1.pack()

def F_M2():
        GPIO.output(12, GPIO.HIGH)
	GPIO.output(13, GPIO.LOW)

def R_M2():
        GPIO.output(12, GPIO.LOW)
	GPIO.output(13, GPIO.HIGH)

def S_M2():
        GPIO.output(12, GPIO.LOW)
	GPIO.output(13, GPIO.LOW)

Nut_F_M2 = tk.Button(TK, height = 5,width = 15, text ="F_M2", command = F_M2)
Nut_R_M2 = tk.Button(TK, height = 5,width = 15, text ="R_M2", command = R_M2)
Nut_S_M2 = tk.Button(TK, height = 5,width = 15, text ="S_M2", command = S_M2)

Nut_F_M2.pack()
Nut_R_M2.pack()
Nut_S_M2.pack()

TK.mainloop()

