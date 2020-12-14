import sys
import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import os
import PIL
import PIL.Image

# customize your API through the following parameters
classes_path = './data/labels/classes.names'
weights_path = './weights/yolov3-tiny.tf'
tiny = True  # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
# path to output folder where images with detections are saved
output_path = './detections/'
num_classes = 7                # number of classes in model

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')


def get_detections(imgName):
    images = []
    raw_images = []
    inputImage = PIL.Image.open(os.path.join(os.getcwd(), imgName))
    images.append(inputImage)
    image_names = []
    for image in images:
        image_names.append(imgName)
        image.save(os.path.join(os.getcwd(), imgName))
        img_raw = tf.image.decode_image(
            open(imgName, 'rb').read(), channels=3)
        raw_images.append(img_raw)

    num = 0

    # create list for final response
    response = []

    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        raw_img = raw_images[j]
        num += 1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)

        t2 = time.time()
        print('time: {}'.format(t2 - t1))
        coordinatesData = {}
        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(
            output_path + 'detection' + str(num) + '.jpg'))


if __name__ == "__main__":
    cnt_args = len(sys.argv)
    if cnt_args < 2:
        print('Need an image name!')
        exit()

    imgName = sys.argv[1]
    get_detections(imgName)
