import C_DHT
import time


while(1):
	print(C_DHT.readSensor(0))
	time.sleep(1)
