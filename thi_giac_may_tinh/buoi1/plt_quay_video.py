import cv2
import matplotlib.pyplot as plt
import numpy as np

#filename = 'cat.mp4'
cap = cv2.VideoCapture(0)

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

fig, ax = plt.subplots(1,1)


#Setup a dummy path
x = np.linspace(0,width,frames)
y = x/2. + 100*np.sin(2.*np.pi*x/1200)

for i in range(frames):
    fig.clf()
    flag, frame = cap.read()

    plt.imshow(frame)
    plt.plot(x,y,'k-', lw=2)
    plt.plot(x[i],y[i],'or')

    if cv2.waitKey(1) == 27:
        break
plt.ion()
plt.show()
