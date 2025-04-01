# FESTEVAL (FUNSD TESSERACT LEVENSHTEIN EVALUATION)

# Tesseract Evaluation on the FUNSD Dataset using Levenshtein Distance

import os
import json
import re
import statistics
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageOps
from Levenshtein import distance, ratio
from pytesseract import image_to_string

dir_path = os.path.dirname(os.path.realpath(__file__))

def perform_eval(annotation, image):
    """
    Evaluates annotation and image pair and yields scores
    The inputs must be a path to the annotation file, and another path to the image file
    """

    filename = os.path.splitext(os.path.basename(annotation))[0]
    
    annotation_path = os.path.join(dir_path, annotation)
    image_path = os.path.join(dir_path, image)

    # Open JSON file
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
        
    form_data = annotation_data['form']

    # Open Image file
    with Image.open(image_path) as img:

        for field in form_data:
            # Get field text
            field_text = field['text'].strip()

            # Skip empty fields
            if field_text == '':
                continue

            # Get bbox xyxy
            xyxy = field['box']

            ocr_text = get_ocr_text(img, xyxy)

            # Skip empty scans
            if ocr_text == '':
                continue

            score = get_score(field_text, ocr_text)

            yield score, ocr_text, field, annotation, image



def get_ocr_text(img, xyxy):

    # Get cropped image
    field_img = img.crop(xyxy)

    # Preprocess image
    # Convert to ndarray
    field_img = np.array(field_img)
    # Resize
    field_img = cv2.resize(field_img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    # Convert to grayscale if needed
    if len(field_img.shape) != 2:
        field_img = cv2.cvtColor(field_img, cv2.COLOR_RGB2GRAY)
    # Denoise
    field_img = cv2.fastNlMeansDenoising(field_img, field_img, h=3, templateWindowSize=7, searchWindowSize=17)
    # Add border
    field_img = cv2.copyMakeBorder(field_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))


    # Get text with ocr
    ocr_text = image_to_string(field_img).strip()

    return ocr_text



def get_score(field_text, ocr_text):

    # Replace all whitespaces with space
    field_text = re.sub(r'[\s\r\n]+', ' ', field_text)
    ocr_text = re.sub(r'[\s\r\n]+', ' ', ocr_text)

    # Remove unsupported characters
    field_text = re.sub(r'[^\w\s\r\n\~!@#$%^&*()_+`\-=[\]\\{}\|;\':",./<>?]', '', field_text)
    ocr_text = re.sub(r'[^\w\s\r\n\~!@#$%^&*()_+`\-=[\]\\{}\|;\':",./<>?]', '', ocr_text)

    # FOR FUNSD SPECIFICALLY
    # Remove spaces before and after certain symbols
    field_text = re.sub(r' ?[\-/] ?', r'-', field_text)
    ocr_text = re.sub(r' ?[\-/] ?', r'-', ocr_text)

    print('ORIGINAL-: ' + field_text)
    print('TESSERACT: ' + ocr_text)

    # Get Levenshtein normalized similarity 1
    # Calculated by: 1 - ldist / lensum 
    # wherein:
    #   lensum = sum(len(a), len(b))
    #   ldist = levenshtein distance where replacements have a cost of 2
    score = ratio(field_text, ocr_text)

    return score



def get_annotations_and_images(data_folders):
    """Yields annotation file path and image file path pairs"""

    for folder in data_folders:

        folder_path = os.path.join(dir_path, folder)

        # Get annotation files
        annotations_path = os.path.join(folder_path, 'annotations')
        annotations = [
            os.path.join(folder, 'annotations', f) for f in os.listdir(annotations_path)
            if f.lower().endswith('.json') and os.path.isfile(os.path.join(annotations_path, f))
        ]

        images_path = os.path.join(folder_path, 'images')

        # For each annotation file:
        for annotation in annotations:
            filename = os.path.splitext(os.path.basename(annotation))[0]

            # Get image
            image = filename + '.jpg'
            if not os.path.isfile(os.path.join(images_path, image)):
                image = filename + '.jpeg'
            if not os.path.isfile(os.path.join(images_path, image)):
                image = filename + '.png'
            if not os.path.isfile(os.path.join(images_path, image)):
                print('No image file found for: ' + annotation)
                continue

            image = os.path.join(folder, 'images', image)

            yield annotation, image



def worker(data):
    """Worker for multiprocessing"""

    annotation, image = data

    results = perform_eval(annotation, image)
    
    return list(results)

def main():
    data_folders = [
        './funsd/training_data',
        './funsd/testing_data'
    ]

    # Process annotations and images using multiprocessing.Pool
    annotations_and_images = get_annotations_and_images(data_folders)

    num_processes = max(1, int(multiprocessing.cpu_count()/2))

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(worker, annotations_and_images)
    results = [r for result in results for r in result]

    # Get average scores
    scores = [result[0] for result in results]
    mean_score = statistics.mean(scores)
    print('Mean Similarity Score:', mean_score)

    # Tabulate and save results
    datetimestr = datetime.today().strftime('%Y-%m-%d %H;%M;%S')

    with open('festeval-results-{0}-all.csv'.format(datetimestr), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            'annotation_file',
            'image_file',
            'field_id',
            'field_text',
            'ocr_text',
            'similarity_score'
        ])

        for result in results:
            score, ocr_text, field, annotation, image = result

            writer.writerow([
                annotation,
                image,
                field['id'],
                field['text'],
                ocr_text,
                score
            ])

    # Save summary
    with open('festeval-results-{0}-summary.csv'.format(datetimestr), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow([
            'no_of_fields_scanned',
            'mean_similarity_score'
        ])

        writer.writerow([
            len(results),
            mean_score
        ])

if __name__ == '__main__':
    main()