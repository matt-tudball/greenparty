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
from sys import exit


TEMP_DIR = "temp"
TEMP_DIR_IMPROC = TEMP_DIR + '\\' + "improc"
PDF2IMAGE_DPI = 200
REGISTER_KEY_STRING = "Poling Station Register"
CSV_HEADER = "id,marked,code,file"
RESULT_FILE = "results.csv"


def init_temp_folder():
    try:
        shutil.rmtree(TEMP_DIR)
    except (OSError, FileNotFoundError) as e:
        print(e)
    print("creating temporary folder...")
    pathlib.Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(TEMP_DIR_IMPROC).mkdir(parents=True, exist_ok=True)


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


def mask_data_to_ignore(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # cv2.imshow('edges', edges)
    minLineLength = 10
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=1, lines=np.array([]),
                            minLineLength=minLineLength, maxLineGap=500)
    a, b, c = lines.shape
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(a):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        if x1 < img.shape[0]/1.9 or x2 < img.shape[0]/1.9:
            continue
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        cv2.line(mask, (x1, x2), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=10)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=10)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        cv2.rectangle(img, (x, 0), (x + w, 0 + int(img.shape[0]/1)), (255, 255, 255), thickness=cv2.FILLED)
        x = x - 201
        cv2.rectangle(img, (x, 0), (x + w, 0 + int(img.shape[0] / 1.1)), (255, 255, 255), thickness=cv2.FILLED)
        cv2.rectangle(img, (0, 0), (0 + int(img.shape[1]), 0 + int(img.shape[0] / 7)), (255, 255, 255), thickness=cv2.FILLED)

    # cv2.imshow('frame', img)
    # cv2.imshow('mak', mask)
    # cv2.waitKey(0)
    # cv2.waitKey(150)
    outpath = "%s\\%s_masked.jpg" % (TEMP_DIR_IMPROC, image_path.split('\\')[-1].split('.')[0])
    cv2.imwrite(outpath, img)


def pre_process_images():
    print("pre processing images...")
    images = glob.glob("%s\\*.jpg" % TEMP_DIR)
    print("found %d images in temp folder." % (len(images)))
    print("rotation skew correction...")
    # for image_path in tqdm(images, total=len(images), initial=1):
    for image_path in images:
        correct_skew(image_path)
        mask_data_to_ignore(image_path)
    print('\t')


def images_to_text():
    print("processing images...")
    images = glob.glob("%s\\*masked.jpg" % TEMP_DIR_IMPROC)
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
    if REGISTER_KEY_STRING in text:
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
    print("start...")
    init_result_file()
    files = glob.glob("C:\\greenparty\\*.pdf")
    n_files = len(files)
    print("found %d files in folder:\n%s" % (n_files, "\n".join(files)))
    init_temp_folder()
    pre_process_file(files)
    pdf_to_images()
    pre_process_images()
    images_to_text()