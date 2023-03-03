
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
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from copy import deepcopy
from numpy import random
import time

# path to test dataset
image_source = "F:/projects/automatic-number-plate-recognition/yolov7/ANPR-1/test/images"
# pat to save outpu images
savepath = "F:/projects/automatic-number-plate-recognition/yolov7/sidebar/v2"
# path to first yolo model
weights = 'F:/projects/automatic-number-plate-recognition/yolov7/yolov7/runs/train/exp/weights/best.pt'
# path to second yolo model
char_recognition_weights = 'F:/projects/automatic-number-plate-recognition/yolov7/yolov7/runs/train/exp1/weights/best.pt'
device_id = 'cpu'
image_size = 640


# Initialize
device = select_device(device_id)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(image_size, s=stride)  # check img_size

char_model = attempt_load(char_recognition_weights,
                          map_location=device)  # load FP32 model
char_stride = int(char_model.stride.max())  # model stride
char_imgsz = check_img_size(image_size, s=char_stride)  # check img_size

#  Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
char_names = char_model.module.names if hasattr(
    model, 'module') else char_model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in char_names]


if half:
    model.half()  # to FP16

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once


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
    labels = []

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
                label = f'{names[int(cls)]} {conf:.2f}'
                plate_detections.append(coords)
                det_confidences.append(conf.item())
                labels.append(label)

    return plate_detections, det_confidences, labels


def crop(image, coord):
    cropped_image = image[int(coord[1]):int(
        coord[3]), int(coord[0]):int(coord[2])]
    return cropped_image


def paste(image1, image2, coords):
    image1[int(coord[1]):int(coord[3]), int(
        coord[0]):int(coord[2])] = image2[:, :]
    return image1


def get_plates_from_image(input):
    if input is None:
        return None
    plate_detections, det_confidences, labels = detect_plate(input)
    detected_image = deepcopy(input)
    detected_plates = []
    for i, coords in enumerate(plate_detections):
        plate_region = crop(input, coords)
        # get the objects in detected plate
        char_detected_plate = detect_chars(plate_region)
        detected_plates.append(char_detected_plate)
        # paste plate image on the original image
        # detected_image = paste(detected_image, char_detected_plate, coords)
        plot_one_box(
            coords, detected_image, label=labels[i], color=[0, 150, 255], line_thickness=1)
    return detected_image, detected_plates


def detect_chars(cropped_plate):
    old_img_w = old_img_h = char_imgsz
    old_img_b = 1

    t0 = time.time()

    img_size = 640
    stride = 32
    # Padded resize
    img = letterbox(cropped_plate, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():
        pred = char_model(img, augment=True)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    t3 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # normalization gain whwh
        gn = torch.tensor(cropped_plate.shape)[[1, 0, 1, 0]]
        s = ''
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], cropped_plate.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # add to string
                s += f"{n} {char_names[int(c)]}{'s' * (n > 1)}, "

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                # label = f'{char_names[int(cls)]} {conf:.2f}'
                label = f'{char_names[int(cls)]}'
                coords = [int(position) for position in (
                    torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                cropped_plate = plot_one_box_PIL(coords, cropped_plate, label=label,
                                                 color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
        print(
            f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Save results (image with detections)
    return cropped_plate


# load all test images
dataset = LoadImages(image_source, img_size=imgsz, stride=stride)
for path, img, plate_image, vid_cap in dataset:
    car_image, plates = get_plates_from_image(plate_image)
    p, s, im0, frame = path, '', plate_image, getattr(dataset, 'frame', 0)
    p = Path(p)  # to Path
    cv.imwrite(os.path.join(savepath, p.name), car_image)
    for i, char_detected_im in enumerate(plates):
        im_name = p.name[0:-3] + '-' + str(i) + '.jpg'
        cv.imwrite(os.path.join(savepath, im_name), char_detected_im)
