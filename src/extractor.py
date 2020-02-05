import glob
import os

import cv2
import shutil
import pathlib
import numpy as np
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
from PyPDF2 import PdfFileWriter, PdfFileReader
import time
from scipy.ndimage import interpolation as inter
from tqdm import tqdm
from skimage import io
from skimage.transform import rotate
from skimage.color import rgb2gray
from deskew import determine_skew
import pytesseract
import re
import imutils
from sys import exit
from numpy import ones, vstack
from numpy.linalg import lstsq
import math
import tesserocr as tr
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


CHAR_TRAINING_DIR = "training"
TEMP_DIR = "temp"
TEMP_DIR_IMPROC = TEMP_DIR + '\\' + "improc"
TEMP_DIR_IMPROC_ROWS = TEMP_DIR_IMPROC + '\\' + "rows"
PDF2IMAGE_DPI = 200
MASK_OFFSET = 800
REGISTER_KEY_STRING_1 = "Station Register"
REGISTER_KEY_STRING_2 = "Station"
REGISTER_KEY_STRING_3 = "Register"
CSV_HEADER = "id,marked,code,file"
RESULT_FILE = "results.csv"


def init_temp_folder():
    try:
        shutil.rmtree(TEMP_DIR)
    except (OSError, FileNotFoundError) as e:
        print(e)
    print("creating temporary folder...")
    pathlib.Path(CHAR_TRAINING_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TEMP_DIR_IMPROC).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TEMP_DIR_IMPROC_ROWS).mkdir(parents=True, exist_ok=True)


def pre_process_file(files):
    print("pdf preprocessing...")
    try:
        for n, file in enumerate(files):
            inputpdf = PdfFileReader(open(file, "rb"))
            for i in range(inputpdf.numPages):
                output = PdfFileWriter()
                output.addPage(inputpdf.getPage(i))
                input_file_name = file.split("\\")[-1].split('.')[0]
                with open("%s\\%s_%06d.pdf" % (TEMP_DIR, input_file_name, i), "wb") as outputStream:
                    output.write(outputStream)
    except PermissionError as e:
        print(e)


def pdf_to_images():
    print("converting pdf to images...")
    files = glob.glob("%s\\*.pdf" % TEMP_DIR)
    try:
        for i, file in tqdm(enumerate(files), total=len(files), initial=1):
            # print("%d/%d progress..." % (i, len(files)))
            try:
                pages = convert_from_path(file, PDF2IMAGE_DPI)
                for j, page in enumerate(pages):
                    out_path = file.split('\\')[-1].split('.')[0]
                    page.save("%s\\%03d_%s.jpg" % (TEMP_DIR, j, out_path), "JPEG")
            except FileNotFoundError as e:
                print("error! could not find file %s." % file, e)
        print('\n')
    except MemoryError as e:
        print("error! not enough memory to process file!", e)


def mask_data_to_ignore(image_path):
    image = io.imread(image_path)
    rotated = np.array(image, dtype=np.uint8)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    ret, thresh = cv2.threshold(gray, 127, 255, 1)

    contours, h = cv2.findContours(thresh, 1, 2)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if cnt[0][0][1] > image.shape[0]/2:
            continue

        area = cv2.contourArea(cnt)
        if area < 15000:
            continue

        if len(approx) == 4:
            cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)


    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_mask, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 2  # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(mask) * 0  # creating a blank to draw lines on
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    if lines is not None:

        for line in lines:
            for x1, y1, x2, y2 in line:
                points = [(x1, y1), (x2, y2)]
                if x1 > 600 and x1 < 1200:
                    continue
                x_coords, y_coords = zip(*points)
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords, rcond=None)[0]

                x1_, y1_ = x1, int(m*(x1 - 90000) + c)
                x2_, y2_ = x2, int(m*(x2 + 90000) + c)
                angle = np.arctan2(y2_ - y1_, x2_ - x1_) * 180. / np.pi

                if int(angle) != 90:
                    continue
                x1_ = x1_ - 40
                x2_ = x2_ - 40
                if x1 < image.shape[0]/2:
                    cv2.line(line_image, (x1_, y1_), (x2_, y2_), (0, 0, 255), 5)
                    cv2.line(image, (x1, y1_), (x2_, y2_), (0, 0, 255), 5)
                    for offset in range(0, 450):
                        cv2.line(line_image, (x1_+offset, y1_), (x2_+offset, y2_), (255, 255, 255), 5)
                        cv2.line(image, (x1_+offset, y1_), (x2_+offset, y2_), (255, 255, 255), 5)
                else:
                    # cv2.line(line_image, (x1_, y1_), (x2_, y2_), (0, 0, 255), 5)
                    # cv2.line(image, (x1_, y1_), (x2_, y2_), (0, 0, 255), 5)
                    x1_ = x1_ + 20
                    x2_ = x2_ + 20
                    offset = 680
                    cv2.line(line_image, (x1_-offset, y1_), (x2_-offset, y2_), (255, 255, 255), 5)
                    cv2.line(image, (x1_-offset, y1_), (x2_-offset, y2_), (255, 255, 255), 5)
                    for offset in range(250, 680):
                        cv2.line(line_image, (x1_-offset, y1_), (x2_-offset, y2_), (255, 255, 255), 5)
                        cv2.line(image, (x1_-offset, y1_), (x2_-offset, y2_), (255, 255, 255), 5)

                # cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    mask = cv2.addWeighted(mask, 0.8, line_image, 1, 0)

    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if cnt[0][0][1] > image.shape[0]/2:
            continue

        area = cv2.contourArea(cnt)
        if area < 15000:
            continue

        if len(approx) == 4:
            cv2.drawContours(mask, [cnt], 0, (0, 0, 0), -1)

    cv2.rectangle(mask, (0, 0), (0 + int(mask.shape[1]), int(mask.shape[0] / 5.7)-20), (0, 0, 0), thickness=cv2.FILLED)

    cv2.rectangle(mask, (0, int(mask.shape[0]/1.1)), (int(mask.shape[1]), int(mask.shape[0]/1.1)+1600), (0, 0, 0), thickness=cv2.FILLED)

    # mask_s = cv2.resize(mask, (640, 820))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    result = gray.copy()
    result[mask == 0] = 255

    rows1, rows2 = find_rows(result, image_path)

    # img_s = cv2.resize(result, (640, 820))
    # cv2.imshow('result', img_s)
    # cv2.imshow('mask', mask_s)
    #
    # cv2.waitKey(0)

    outpath = "%s\\%s_masked.jpg" % (TEMP_DIR_IMPROC, image_path.split('\\')[-1].split('.')[0])
    io.imsave(outpath, result.astype(np.uint8))

    outpath = "%s\\%s_rows1.jpg" % (TEMP_DIR_IMPROC, image_path.split('\\')[-1].split('.')[0])
    io.imsave(outpath, rows1.astype(np.uint8))

    outpath = "%s\\%s_rows2.jpg" % (TEMP_DIR_IMPROC, image_path.split('\\')[-1].split('.')[0])
    io.imsave(outpath, rows2.astype(np.uint8))
    return outpath


def find_rows(result, image_path):
    cv_img = result.copy()
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    x1, y1 = 0, 0
    x2, y2 = 500, cv_img.shape[0]

    cv_img_1 = cv_img[y1:y2, x1:x2]
    cv_img_2 = cv_img[y1:y2, x1+700:x2+700]

    cv_img_1_o = cv_img_1.copy()
    cv_img_2_o = cv_img_2.copy()
    api = tr.PyTessBaseAPI(path="C:\Program Files (x86)\Tesseract-OCR")
    try:
        pil_img = Image.fromarray(cv2.cvtColor(cv_img_1, cv2.COLOR_BGR2RGB))
        api.SetImage(pil_img)
        boxes = api.GetComponentImages(tr.RIL.TEXTLINE, False)
        text = api.GetUTF8Text()
        for i, (im, box, _, _) in enumerate(boxes):
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            if h < 17 or h > 40:
                continue
            x = 0
            w = cv_img_1.shape[1]-10
            cv2.rectangle(cv_img_1, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            outpath = "%s\\%s_left_%03d.jpg" % (TEMP_DIR_IMPROC_ROWS, image_path.split('\\')[-1].split('.')[0], i)
            roi = cv_img_1_o[y-5:y+h, x:x+w+5]
            io.imsave(outpath, roi.astype(np.uint8))

        pil_img = Image.fromarray(cv2.cvtColor(cv_img_2, cv2.COLOR_BGR2RGB))
        api.SetImage(pil_img)
        boxes = api.GetComponentImages(tr.RIL.TEXTLINE, False)
        text = api.GetUTF8Text()
        for i, (im, box, _, _) in enumerate(boxes):
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            if h < 20 or h > 35:
                continue
            x = 0
            w = cv_img_1.shape[1]-10
            cv2.rectangle(cv_img_2, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
            outpath = "%s\\%s_right_%03d.jpg" % (TEMP_DIR_IMPROC_ROWS, image_path.split('\\')[-1].split('.')[0], i)
            roi = cv_img_2_o[y-5:y+h, x:x+w+5]
            io.imsave(outpath, roi.astype(np.uint8))
    finally:
        api.End()

    # cv_img1_s = cv2.resize(cv_img_1, (240, 820))
    # cv2.imshow('cv_img1_s', cv_img1_s)
    # cv_img2_s = cv2.resize(cv_img_2, (240, 820))
    # cv2.imshow('cv_img2_s', cv_img2_s)
    return cv_img_1, cv_img_2


def correct_skew(image_path):
    image = io.imread(image_path)
    grayscale = rgb2gray(image)
    angle = determine_skew(grayscale)
    if angle < -10:
        angle = angle + 90
    rotated = rotate(image, angle, resize=True) * 255
    outpath = "%s\\%s" % (TEMP_DIR_IMPROC, image_path.split('\\')[-1])
    io.imsave(outpath, rotated.astype(np.uint8))
    return outpath


def pre_process_images():
    print("pre processing images...")
    images = glob.glob("%s\\*.jpg" % TEMP_DIR)
    print("found %d images in temp folder." % (len(images)))
    # for image_path in tqdm(images, total=len(images), initial=1):
    for image_path in images:
        image_path_deskew = correct_skew(image_path)
        mask_data_to_ignore(image_path_deskew)
    print('\t')


def extract_chars(image_path):
    print(0)


def build_training_set_from_rows():
    print("building training sets...")
    images = glob.glob("%s\\*.jpg" % CHAR_TRAINING_DIR)
    print("found %d images in rows folder." % (len(images)))
    # for image_path in tqdm(images, total=len(images), initial=1):
    for image_path in images:
        extract_chars(image_path)
    print('\t')


def images_to_text():
    print("processing images...")
    images = glob.glob("%s\\_masked*.jpg" % TEMP_DIR_IMPROC)
    print("found %d preprocessed images." % (len(images)))
    print("looking for text...")
    for image_path in tqdm(images, total=len(images), initial=1):
        config = ('-l eng --oem 1 --psm 3')
        im = cv2.imread(image_path, cv2.IMREAD_COLOR)
        text = pytesseract.image_to_string(im, config=config)
        output_text_file_path = image_path.split('.jpg')[0] + ".txt"
        with open(output_text_file_path, "w") as text_file:
            text_file.write(text)
        text_to_csv(text, output_text_file_path)
    print('\t')


def clean_address(string):
    cleaned = re.sub("Flat \d+", '', string)
    cleaned = re.sub("Flat\d+", '', cleaned)
    cleaned = re.sub("Gate \d+", '', cleaned)
    cleaned = re.sub("Gate\d+", '', cleaned)
    cleaned = re.sub("House \d+", '', cleaned)
    cleaned = re.sub("House\d+", '', cleaned)
    cleaned = re.sub("Maisonette \d+", '', cleaned)
    cleaned = re.sub("Maisonette \d+", '', cleaned)
    cleaned = re.sub("Court \d+", '', cleaned)
    cleaned = re.sub("Court\d+", '', cleaned)
    return cleaned


def text_to_csv(text, text_file_path):
    # if REGISTER_KEY_STRING_1 in text:
    with open(text_file_path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content_filtered = []
    chars = set('0123456789-~_')
    for item in content:
        if len(item) == 0:
            continue
        if any((c in chars) for c in item[0]):
            data = clean_address(item)
            input_file_id = text_file_path.split('\\')[-1].split('.')[0].split('_')[1]
            id, marked, code = parse_row_data(data)
            append_result_file(RESULT_FILE, id, marked, code, input_file_id)


def parse_row_data(data):
    print(data)
    return 0, False, ''


def init_result_file():
    try:
        os.remove(RESULT_FILE)
    except OSError:
        pass
    with open(RESULT_FILE, 'a') as outfile:
        outfile.write(CSV_HEADER)
        outfile.write('\n')
    outfile.close()
    return RESULT_FILE


def append_result_file(filename, id, marked, code, file):
    data = "%d,%s,%s,%s" % (id, marked, code, file)
    with open(filename, 'a') as outfile:
        outfile.write(data)
        outfile.write('\n')
    outfile.close()


if __name__ == '__main__':
    # print("start...")
    # init_result_file()
    # files = glob.glob("C:\\greenparty\\*.pdf")
    # n_files = len(files)
    # print("found %d files in folder:\n%s" % (n_files, "\n".join(files)))
    # init_temp_folder()
    # pre_process_file(files)
    # pdf_to_images()
    pre_process_images()
    images_to_text()