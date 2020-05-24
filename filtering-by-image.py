import cv2
import time
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


path = (input("Enter the Path of File:"))
img = cv2.imread(path, 1)

dict_1 = {
    1: 'Binary image',
    2: 'Canny edge detection',
    3: 'Focus circle blur',
    4: 'Stylization Filter',
    5: 'Sepia Filter',
    6: 'Warming Filter',
}

height = img.shape[0]
width = img.shape[1]
for i in range(1, len(dict_1)+1):
    print(i, '. ', dict_1.get(i))
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
        replaced_image = cv2.stylization(img, sigma_s=50, sigma_r=0.05)
        cv2.imshow("Stylization Filter", replaced_image)
        img1 = replaced_image
    elif num == '5':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        replaced_image = cv2.filter2D(img, -1, kernel)
        cv2.imshow("Sepia Filter", replaced_image)
        img1 = replaced_image
    elif num == '6':
        increaseLookupTable = spreadLookupTable(
            [0, 64, 128, 256], [0, 80, 160, 256])
        decreaseLookupTable = spreadLookupTable(
            [0, 64, 128, 256], [0, 50, 100, 256])
        red_channel, green_channel, blue_channel = cv2.split(img)
        red_channel = cv2.LUT(
            red_channel, increaseLookupTable).astype(np.uint8)
        blue_channel = cv2.LUT(
            blue_channel, decreaseLookupTable).astype(np.uint8)
        replaced_image = cv2.merge((red_channel, green_channel, blue_channel))

        cv2.imshow("Warming Filter", replaced_image)
        img1 = replaced_image
    # elif num == '7':
    # elif num == '8':
    # elif num == '9':
    # elif num == '10':
    # elif num == '11':
    # elif num == '12':
    # elif num == '13':
    # elif num == '14':
    # elif num == '15':
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
