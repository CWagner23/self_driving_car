import numpy as np
import cv2

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1*3/5)         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope <= 0:  # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return averaged_lines


def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(gray, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)

    triangle = np.array([[
        (200, height),
        (550, 250),
        (1100, height), ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def line_detector(image):
    lane_image = np.copy(image)
    lane_canny = canny(image)
    cropped_canny = region_of_interest(lane_canny)
    lines = cv2.HoughLinesP(
        cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(lane_image, averaged_lines)
    return cv2.addWeighted(lane_image, 0.8, line_image, 1, 0)

def detect_lines_image(inputImage):
    image = cv2.imread(inputImage)
    combo_image = line_detector(image)
    cv2.imshow("image", combo_image)
    cv2.waitKey(0)

def detect_lines_video(video):
    # press q to exit video
    
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        combo_image = line_detector(frame)
        cv2.imshow("image", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# detect_lines_image('Data/test_image.jpg')
detect_lines_video("Data/test2.mp4")
