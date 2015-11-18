# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2


def load_image_from_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image to be scanned")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    return image


def resize_image(image, desired_height):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / desired_height
    orig = image.copy()
    image = imutils.resize(image, height=desired_height)
    return image, orig, ratio


def image_to_grey_blur_canny_edges(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return edged


def edged_image_to_contours(edged):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            return approx
            # screenCnt = approx
            # break
    return -1


def image_to_scan_bird_style_view(image, screenCnt, ratio):
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)
    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped = threshold_adaptive(warped, 250, offset=10)
    warped = warped.astype("uint8") * 255
    return warped


def save_scanned_image():
    image = load_image_from_args()
    image, orig, ratio = resize_image(image, 500)
    edged = image_to_grey_blur_canny_edges(image)
    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    screenCnt = edged_image_to_contours(edged)

    # show the contour (outline) of the piece of paper
    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    warped = image_to_scan_bird_style_view(orig, screenCnt, ratio)

    # show the original and scanned images
    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height=650))
    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    # cv2.imwrite("images/morse_scanned.jpg", warped)
    cv2.waitKey(0)


if __name__ == '__main__':
    letters_to_morse = {'A': '.-', 'B': '-...', 'C': '-.-.',
                        'D': '-..', 'E': '.', 'F': '..-.',
                        'G': '--.', 'H': '....', 'I': '..',
                        'J': '.---', 'K': '-.-', 'L': '.-..',
                        'M': '--', 'N': '-.', 'O': '---',
                        'P': '.--.', 'Q': '--.-', 'R': '.-.',
                        'S': '...', 'T': '-', 'U': '..-',
                        'V': '...-', 'W': '.--', 'X': '-..-',
                        'Y': '-.--', 'Z': '--..',
                        '0': '-----', '1': '.----', '2': '..---',
                        '3': '...--', '4': '....-', '5': '.....',
                        '6': '-....', '7': '--...', '8': '---..',
                        '9': '----.'
                        }
    morse_to_letters = dict((v, k) for k, v in letters_to_morse.items())
    print(morse_to_letters)

    # image = load_image_from_args()
    image = cv2.imread("images/morse_scanned.jpg")
    h, w = image.shape[:2]
    image = image[0.03*h:0.97*h,0.03*w:0.97*w] #crop image to cut away borders with fuzz
    h, w = image.shape[:2]
    # image = image_to_grey_blur_canny_edges(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(1,5):
        image = cv2.GaussianBlur(image, (9, 9), 0)
        _, image = cv2.threshold(image, 170, 255, 0)
    # image = cv2.Canny(image, 75, 200)
    # morse = cv2.dilate(image,kernel,iterations = 1)
    # edged = cv2.Canny(gray, 75, 200)
    morse, cnts, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    maxContourLenght = (2*h+2*w) * 0.5
    morse_cnts = []
    for c in cnts:
        if (len(c) > 30) and (cv2.arcLength(c,1) < maxContourLenght):
            morse_cnts.append(c)

    # print(morse_cnts)
    cv2.drawContours(image, morse_cnts, -1, (100, 120, 0), 10)
    # morse = cv2.morphologyEx(morse, cv2.MORPH_OPEN, kernel)
    # morse = cv2.dilate(image,kernel,iterations = 1)
    # gray = cv2.cvtColor(morse, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(morse, (5, 5), 0)
    # edged = cv2.Canny(gray, 75, 200)
    # morse = cv2.Canny(image, 180, 200)

    cv2.imshow("Scanned", imutils.resize(image, height=650))
    cv2.waitKey(0)
    # save_scanned_image()
