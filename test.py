import cv2
import numpy as np

from network.mobilenetv2 import mobilenet_v2

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

MODEL_PATH = 'checkpoints/10-17.06.h5'

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
    # model.summary()

    model.load_weights(MODEL_PATH, by_name=True)

    origin_img = cv2.imread('dataset/000026.jpg')
    h, w, _ = origin_img.shape
    img = cv2.resize(origin_img, INPUT_IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125

    result = model.predict(np.expand_dims(img, axis=0))[0]

    for res in result:
        if res[0] == 0: continue  # background
        if res[1] < 0.5: continue  # score

        min_x, min_y, max_x, max_y = res[2:]
        cv2.rectangle(origin_img, (int(min_x * w), int(min_y * h)), (int(max_x * w), int(max_y * h)), (0, 255, 0), 2)
        cv2.putText(origin_img, f'{classes[int(res[0])]}_{res[1]:.2f}', (int(min_x * w), int(min_y * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite('dataset/output.jpg', origin_img)


if __name__ == '__main__':
    main()
