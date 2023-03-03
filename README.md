# automatic-number-plate-recognition
Using YOLOv7 to recognize Iraning number plates in images

1. training a YOLOv7 on images containing number plates to detect plate region
2. v1 -> using easyocr to read number plates <br>
   v2 -> training another YOLOv7 on number plate images [dataset](https://github.com/mut-deep/IR-LPR) labeled to detect characters 
## Implemetation steps and Usage guide
1. I combined 2 persian number plates datasets and uploaded them to roboflow to generate labels to create a [dataset](https://universe.roboflow.com/assessment-khz6n/anpr-ca6wi) of images containing number plates, with sufficient size, [dataset1](https://www.kaggle.com/datasets/samyarr/iranvehicleplatedataset), [dataset2](https://www.kaggle.com/datasets/skhalili/iraniancarnumberplate)
2. notebook [number_plate_detection.ipynb](https://github.com/PariaKashani/automatic-number-plate-recognition/blob/main/number_plate_detection.ipynb) contains the codes to train YOLOv7 and detect number plates in images. After running the notebook you will have a traied model and you can use it in next steps.
3. v1: [number-plate-recognition-v1.py](https://github.com/PariaKashani/automatic-number-plate-recognition/blob/main/number-plate-recognition-v1.py) runs the model on the test set and then passes the detected number plates to easyocr to read the number plate. This approach doesn't generate accurate predictions. you can check the output images in this [folder](https://github.com/PariaKashani/automatic-number-plate-recognition/tree/main/sidebar/ocr-on-cars-test-v1). (notice: to run this code you need to copy [plots.py](https://github.com/PariaKashani/automatic-number-plate-recognition/blob/main/plots.py) and [yekan.ttf](https://github.com/PariaKashani/automatic-number-plate-recognition/blob/main/yekan.ttf) into utils folder of yolov7)
4. v2: notebook [number_plate_ocr.ipynb](https://github.com/PariaKashani/automatic-number-plate-recognition/blob/main/number_plate_ocr.ipynb) trains another YOLOv7 to detect and recognize characters in a number plate.
5. v2: [number-plate-recognition-v2.py](https://github.com/PariaKashani/automatic-number-plate-recognition/blob/main/number-plate-recognition-v2.py) uses both traied models. First model detects the number plate region. I cropped the region and passed the new image to second model. The accuracy of this method is amazing. You can check the output in this [folder](https://github.com/PariaKashani/automatic-number-plate-recognition/tree/main/sidebar/ocr-on-cars-test-v2). for each input you will have 2 or more output images based on the number of number plates detected in the image.


