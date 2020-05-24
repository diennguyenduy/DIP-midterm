import cv2
import time
import numpy as np
import scipy.ndimage
from scipy.interpolate import UnivariateSpline


def spreadLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def grayscale(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def dodge(front, back):
    result = front*255/(255-(back-1))
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')


def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=-
                       1)  # detects horizontal edges
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=-
                       1)  # detects vertical edges
    # combine both edges
    return cv2.bitwise_or(opImgx, opImgy)


def resize(image, window_height=500):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height/aspect_ratio
    image = cv2.resize(image, (int(window_height), int(window_width)))
    return image


def alpha_blend(frame_1, frame_2, mask):
    alpha = mask/255.0
    blended = cv2.convertScaleAbs(frame_1*(1-alpha) + frame_2*alpha)
    return blended


path = (input("Enter the Path of File:"))
img = cv2.imread(path, 1)

dict_1 = {
    1: 'Binary image',
    2: 'Canny edge detection',
    3: 'Focus circle blur',
    4: 'Stylization Filter',
    5: 'Sepia Filter',
    6: 'Warming Filter',
    7: 'Convert to sketch',
    8: 'Saturation',
    9: 'Green',
    10: 'Cartoon image',
    11: 'Pencil',
    12: 'Snow',
    13: 'Laplacian',
    14: 'Bilateral',
    15: 'Value channel'
}

height = img.shape[0]
width = img.shape[1]
for i in range(1, len(dict_1)+1):
    print(i, '. ', dict_1.get(i))
while(True):
    num = input('Enter the number which type of image you want: ')
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
    elif num == '7':
        g = grayscale(img)
        i = 255-g

        b = scipy.ndimage.filters.gaussian_filter(i, sigma=3)
        r = dodge(b, g)

        edges = cv2.Canny(img, 100, 200)
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_clone = np.zeros_like(edges)
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size > 10:
                cv2.drawContours(r, [cnt], -1, (200, 0, 50), 1)

        cv2.imshow('Sketch', cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
    elif num == '8':
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("Saturation", img_HSV[:, :, 1])
        img1 = img_HSV[:, :, 1]
    elif num == '9':
        B, G, R = cv2.split(img)
        zeros = np.zeros((height, width), dtype="uint8")
        cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
        img1 = cv2.merge([zeros, G, zeros])
    elif num == '10':
        img_color = img
        for _ in range(2):
            img_color = cv2.pyrDown(img_color)
        for _ in range(5):
            img_color = cv2.bilateralFilter(
                img_color, d=9, sigmaColor=9, sigmaSpace=7)
        # upsample image to original size
        for _ in range(2):
            img_color = cv2.pyrUp(img_color)
        img_color = cv2.resize(img_color, img.shape[:2])
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        img_edge = cv2.adaptiveThreshold(img_blur,
                                         maxValue=255,
                                         adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                         thresholdType=cv2.THRESH_BINARY,
                                         blockSize=9,
                                         C=3)
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_edge = cv2.resize(img_edge, img.shape[:2])
        img_cartoon = cv2.bitwise_and(img_color, img_edge)
        cv2.imshow('Cartoon image', img_cartoon[:, :, ::-1])
    elif num == '11':
        # Blur it to remove noise
        frame = cv2.GaussianBlur(img, (3, 3), 0)
        # make a negative image
        invImg = 255-frame

        # Detect edges from the input image and its negative
        edgImg0 = sobel(frame)
        edgImg1 = sobel(invImg)
        # different weights can be tried too
        edgImg = cv2.addWeighted(edgImg0, 1, edgImg1, 1, 0)

        # Invert the image back
        res = 255-edgImg
        res = np.roll(res, res.shape[1]//2, axis=1)
        img1 = res
        cv2.imshow('Pencil', res)
    elif num == '12':
        image_HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # Conversion to HLS
        image_HLS = np.array(image_HLS, dtype=np.float64)
        brightness_coefficient = 2.5
        snow_point = 100  # increase this for more snow
        image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = image_HLS[:,
                                                                        :, 1][image_HLS[:, :, 1] < snow_point]*brightness_coefficient
        # scale pixel values up for channel 1(Lightness)
        image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
        # Sets all values above 255 to 255
        image_HLS = np.array(image_HLS, dtype=np.uint8)
        image_RGB = cv2.cvtColor(
            image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB
        img1 = image_RGB
        cv2.imshow("Snow", image_RGB)
    elif num == '13':
        new_image = cv2.Laplacian(img, cv2.CV_64F)
        cv2.imshow('Laplacian', new_image)
    elif num == '14':
        # 9 ,75 and 75 are sigma color value and sigma space value affects cordinates space and color space
        bilateral = cv2.bilateralFilter(img, 7, 20, 20)
        cv2.imshow("Bilateral", bilateral)
        img1 = bilateral
    elif num == '15':
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("Value channel", img_HSV[:, :, 2])
        img1 = img_HSV[:, :, 2]
    else:
        print('invalid input!')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save = input('Do you want to save?(y/n) ')
    if save == 'y':
        file = input('Enter the image name to be saved: ')
        cv2.imwrite(file+'.jpg', img1)
        break
    else:
        break
