import cv2
import numpy as np

from network.mobilenetv2 import mobilenet_v2

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

MODEL_PATH = 'checkpoints/39-9.46.h5'

BATCH_SIZE = 32
INPUT_IMG_SIZE = (300, 300)
ASPECT_RATIOS = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]
LAYERS_DEPTHS = [-1, -1, 512, 256, 256, 128]
SCALES_PASCAL = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]


def main():
    model = mobilenet_v2(INPUT_IMG_SIZE, len(classes), ASPECT_RATIOS, LAYERS_DEPTHS, SCALES_PASCAL, 'inference')
    model.summary()

    model.load_weights(MODEL_PATH, by_name=True)

    img = cv2.imread('dataset/VOCdevkit/VOC2007/JPEGImages/000022.jpg')
    img = cv2.resize(img, INPUT_IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125

    result = model.predict(np.expand_dims(img, axis=0))

    print(classes[int(result[0,0,0])])


if __name__ == '__main__':
    main()
