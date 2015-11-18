# USAGE
# python scan.py --image images/page.jpg

# import the necessary packages
from pyimagesearch.transform import four_point_transform
from pyimagesearch import imutils
from skimage.filters import threshold_adaptive
import math
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
    # print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    screenCnt = edged_image_to_contours(edged)

    # show the contour (outline) of the piece of paper
    # print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    warped = image_to_scan_bird_style_view(orig, screenCnt, ratio)

    # show the original and scanned images
    # print("STEP 3: Apply perspective transform")
    # cv2.imshow("Original", imutils.resize(orig, height=650))
    # cv2.imshow("Scanned", imutils.resize(warped, height=650))
    # cv2.imwrite("images/morse_scanned.jpg", warped)
    # cv2.waitKey(0)
    # return imutils.resize(warped, height=1200)
    # cv2.imshow("Scanned", imutils.resize(warped, height=650))
    # cv2.waitKey(0)
    return warped, orig


def sort_by_centroids(contours, centroids):
    for i in range(len(contours)):
        for j in range(i, len(contours)):
            if centroids[j][0] < centroids[i][0]:
                centroids[j], centroids[i] = centroids[i], centroids[j]
                contours[j], contours[i] = contours[i], contours[j]
    return contours, centroids


def find_morse_message(contours, centroids):
    result = ''
    for i in range(len(contours)):
        contour_lenght = cv2.arcLength(contours[i], 1)
        r = contour_lenght/(2*math.pi)
        if math.floor(0.7*r) <= cv2.pointPolygonTest(contours[i], (centroids[i][0],centroids[i][1]), True)  <= math.ceil(1.3*r):
            result = result + '.'
        else:
            result = result + '-'
    return result


def translate_letters_to_morse(x):
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
    return letters_to_morse[x]


def translate_morse_to_letters(x):
    morse_to_letters = {'--..': 'Z', '.': 'E', '..-.': 'F', '-.-.': 'C', '-': 'T', '.-.': 'R', '---': 'O', '--.': 'G',
                        '..': 'I', '--...': '7', '....': 'H', '---..': '8', '.----': '1', '-.': 'N', '--.-': 'Q', '..---': '2',
                        '----.': '9', '....-': '4', '-..-': 'X', '-...': 'B', '-----': '0', '.-..': 'L', '...-': 'V', '-..': 'D',
                        '.--': 'W', '..-': 'U','.--.': 'P', '-.-': 'K', '.....': '5', '-....': '6',
                        '-.--': 'Y', '...': 'S', '--': 'M', '...--': '3', '.-': 'A', '.---': 'J'}
    return morse_to_letters[x]


def crop_and_clear_image(image):
    h, w = image.shape[:2]
    image = image[0.03*h:0.97*h,0.03*w:0.97*w] #crop image to cut away borders with fuzz
    h, w = image.shape[:2]
    grey_image = image
    #grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(1,5):
        grey_image = cv2.GaussianBlur(grey_image, (9, 9), 0)
        _, grey_image = cv2.threshold(grey_image, 170, 255, 0)
    # grey_image = cv2.dilate(grey_image, np.ones((5,5), np.uint8))
    # grey_image = cv2.erode(grey_image, np.ones((5,5), np.uint8))
    return grey_image


def find_centroids(image):
    h, w = image.shape[:2]
    morse, cnts, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    morse_cnts = []
    morse_cent = []
    for c in cnts:
        if (len(c) > 30) and (cv2.arcLength(c,1) < ((2*h+2*w) * 0.5)):
            morse_cnts.append(c)
            M = cv2.moments(c)
            morse_cent.append([int(M['m10']/M['m00']), int(M['m01']/M['m00'])]) #create list of centroids
    return morse_cent, morse_cnts


def find_morse_contours(group_morse):
    h, w = group_morse.shape[:2]
    group_morse = cv2.erode(group_morse, np.ones((11,11), np.uint8), iterations=8) #erode to create big contours - groups, aka letters
    _, grp_cnts, _ = cv2.findContours(group_morse.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    morse_grp_cnts = []
    for c in grp_cnts:
        if cv2.arcLength(c,1) < ((2*h+2*w) * 0.5):
            morse_grp_cnts.append(c)

    morse_groups = []
    morse_groups_cent = []
    for grp_cnt in morse_grp_cnts:
        grp = []
        grp_cent = []
        for i in range(0, len(morse_cnts)):
            if cv2.pointPolygonTest(grp_cnt, (morse_cent[i][0], morse_cent[i][1]), False) > 0:
                #if centroid is in group shape, this dot/dash belongs to group
                grp.append(morse_cnts[i])
                grp_cent.append(morse_cent[i])
        morse_groups.append(grp) #add group to group set
        morse_groups_cent.append(grp_cent)

    morse_groups = morse_groups[::-1]
    morse_groups_cent = morse_groups_cent[::-1]

    for i in range(len(morse_groups)):
        morse_groups[i], morse_groups_cent[i] = sort_by_centroids(morse_groups[i], morse_groups_cent[i])

    return group_morse, morse_groups, morse_groups_cent, morse_grp_cnts


def get_morse_text(morse_groups, morse_groups_cent):
    morse_text = []
    for i in range(len(morse_groups)):
        morse_text.append(translate_morse_to_letters(find_morse_message(morse_groups[i],morse_groups_cent[i])))
    return morse_text

if __name__ == '__main__':
    # image = load_image_from_args()
    image, true_orig = save_scanned_image()

    #cv2.imwrite("images/morse_scanned2.jpg", image)
    #image = cv2.imread("images/morse_scanned2.jpg")
    orig = image.copy()
    grey_image = crop_and_clear_image(image)
    cv2.imshow("After cropping", imutils.resize(grey_image, height=650))
    cv2.waitKey(0)
    morse_cent, morse_cnts = find_centroids(grey_image)

    group_morse = grey_image.copy()
    group_morse, morse_groups, morse_groups_cent, morse_grp_cnts = find_morse_contours(group_morse)
    cv2.imshow("Morse code grouped", imutils.resize(group_morse, height=650))
    cv2.waitKey(0)
    morse_text = get_morse_text(morse_groups, morse_groups_cent)
    for i in range(len(morse_groups)):
        center = morse_groups_cent[i][0]
        cv2.putText(orig, morse_text[i], (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0 ,255), 10)

    cv2.imshow("Morse code scanned", imutils.resize(orig, height=650))
    cv2.imshow("Orginal", imutils.resize(true_orig, height=650))
    cv2.waitKey(0)
    # save_scanned_image()