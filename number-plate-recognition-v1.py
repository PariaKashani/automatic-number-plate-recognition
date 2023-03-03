
import os
from pathlib import Path
from typing import Union
import torch
import cv2 as cv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.datasets import LoadStreams, LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box, plot_one_box_PIL
from copy import deepcopy
import easyocr


# path to image in case of processing one image
# images_n_vids_path = "F:/projects/automatic-number-plate-recognition/yolov7/images_vids"
# image_path = os.path.join(images_n_vids_path, "329.jpg")

# path to dataset of test images
image_source = "F:/projects/automatic-number-plate-recognition/yolov7/ANPR-1/test/images"
# path to save outputs
savepath = "F:/projects/automatic-number-plate-recognition/yolov7/sidebar/v1"
weights = 'F:/projects/automatic-number-plate-recognition/yolov7/yolov7/runs/train/exp/weights/best.pt'
device_id = 'cpu'
image_size = 640

# Initialize
device = select_device(device_id)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(image_size, s=stride)  # check img_size


if half:
    model.half()  # to FP16

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once


# Load OCR
reader = easyocr.Reader(['fa'])


def detect_plate(source_image):
    # Padded resize
    img_size = 640
    stride = 32
    img = letterbox(source_image, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        # Inference
        pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=1, agnostic=True)

    plate_detections = []
    det_confidences = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], source_image.shape).round()

            # Return results
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (
                    torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())

    return plate_detections, det_confidences


def crop(image, coord):
    cropped_image = image[int(coord[1]):int(
        coord[3]), int(coord[0]):int(coord[2])]
    return cropped_image


def ocr_plate(plate_region):
    # Image pre-processing for more accurate OCR
    cv.imwrite(os.path.join(savepath, "plate_img.png"), plate_region)
    rescaled = cv.resize(plate_region, None, fx=1.2,
                         fy=1.2, interpolation=cv.INTER_CUBIC)
    grayscale = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    # OCR the preprocessed image
    grayscale_blur = cv.medianBlur(grayscale, 1)
    ret, thresh1 = cv.threshold(
        grayscale_blur, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(os.path.join(savepath, "grayscale_blur.png"), grayscale_blur)
    plate_text_easyocr = reader.readtext(grayscale_blur)
    if plate_text_easyocr:
        (bbox, text_easyocr, ocr_confidence) = plate_text_easyocr[0]
        print("plate_text Easyocr ", text_easyocr)
    else:
        text_easyocr = "_"
        ocr_confidence = 0
    # if ocr_confidence == 'nan':

    return text_easyocr, ocr_confidence


def get_plates_from_image(input):
    if input is None:
        return None
    plate_detections, det_confidences = detect_plate(input)
    plate_texts = []
    ocr_confidences = []
    detected_image = deepcopy(input)
    for coords in plate_detections:
        plate_region = crop(input, coords)
        plate_text, ocr_confidence = ocr_plate(plate_region)
        plate_texts.append(plate_text)
        ocr_confidences.append(ocr_confidence)
        detected_image = plot_one_box_PIL(
            coords, detected_image, label=plate_text, color=[0, 150, 255], line_thickness=2)
    return detected_image


def get_best_ocr(preds, rec_conf, ocr_res, track_id):
    for info in preds:
        # Check if it is current track id
        if info['track_id'] == track_id:
          # Check if the ocr confidenence is maximum or not
            if info['ocr_conf'] < rec_conf:
                info['ocr_conf'] = rec_conf
                info['ocr_txt'] = ocr_res
            else:
                rec_conf = info['ocr_conf']
                ocr_res = info['ocr_txt']
            break
    return preds, rec_conf, ocr_res


# load one image
# plate_image = cv.imread(image_path)
# detected_plate_image = get_plates_from_image(plate_image)
# cv.imwrite(os.path.join(savepath, "detected_plate.png"), detected_plate_image)

# load all test images
dataset = LoadImages(image_source, img_size=imgsz, stride=stride)
for path, img, plate_image, vid_cap in dataset:
    detected_plate_image = get_plates_from_image(plate_image)
    p, s, im0, frame = path, '', plate_image, getattr(dataset, 'frame', 0)
    p = Path(p)  # to Path
    cv.imwrite(os.path.join(savepath, p.name),
               detected_plate_image)
