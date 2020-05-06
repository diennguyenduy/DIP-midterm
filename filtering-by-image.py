import cv2
import time
import numpy as np

path = (input("Enter the Path of File:"))
img = cv2.imread(path, 1)

dict_1 = {
    1: 'Binary image',
    2: 'Detecting the edge of image',
    3: 'Focus circle blur',
    4: 'Face Detection'
}

height = img.shape[0]
width = img.shape[1]
for i in range(1, len(dict_1)+1):
    print(i, '.', dict_1.get(i))
while(True):
    num = input('Enter the number which type of image you want:')
    cv2.imshow('image', img)
    if num == '1':
        # through thresholding we will try to provide the value the value through which we can put below the particular value we assign the value 0 and above it will be white.
        ret, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("Binary", bw)
        img1 = bw
    elif num == '2':
        # This demand two thresholds from us i.e; 20 and 170 this is like lower and upper value
        canny = cv2.Canny(img, 20, 170)
        cv2.imshow("canny", canny)
        img1 = canny
    elif num == '3':
        frame_h, frame_w, frame_c = img.shape
        y = int(frame_h/2)
        x = int(frame_w/2)

        mask = np.zeros((frame_h, frame_w, 3), dtype='uint8')
        cv2.circle(mask, (x, y), int(y/2), (255, 255, 255), -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)

        blured = cv2.GaussianBlur(img, (21, 21), 11)
        alpha = (255-mask)/255.0
        blended = cv2.convertScaleAbs(img*(1-alpha) + blured*alpha)
        frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)

        cv2.imshow('focus', frame)
        img1 = frame
    elif num == '4':
        face_cascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.06, minNeighbors=6)
        for x, y, w, h in faces:
            img1 = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.imshow("Gray", img1)
    else:
        print('invalid input')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save = input('Do you want to save?y/n')
    if save == 'y':
        file = input('Enter the image name to be saved')
        cv2.imwrite(file+'.jpg', img1)
        break
    else:
        break
