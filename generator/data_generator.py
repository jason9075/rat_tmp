import os

import sklearn
from bs4 import BeautifulSoup


class DataGenerator:
    def __init__(self, ssd_encoder):

        self.labels = []
        self.img_file_path = []
        self.ssd_encoder = ssd_encoder

    def parse(self, images_dir, annos_dir, img_set_path, category_name):
        with open(img_set_path) as f:
            image_ids = [line.strip() for line in f]

        for image_id in image_ids:

            self.img_file_path.append(os.path.join(images_dir, '{}.jpg'.format(image_id)))

            with open(os.path.join(annos_dir, '{}.xml'.format(image_id))) as f:
                soup = BeautifulSoup(f, 'lxml')

            boxes = []
            height = float(soup.find('height').text)
            width = float(soup.find('width').text)
            objects = soup.find_all('object')

            for obj in objects:
                class_name = obj.find('name', recursive=False).text
                class_id = category_name.index(class_name)
                bndbox = obj.find('bndbox', recursive=False)
                # 因訓練圖片長寬不一，所以都用比例表示
                xmin = float(bndbox.xmin.text) / width
                ymin = float(bndbox.ymin.text) / height
                xmax = float(bndbox.xmax.text) / width
                ymax = float(bndbox.ymax.text) / height

                boxes.append([class_id, xmin, ymin, xmax, ymax])

            self.labels.append(boxes)

    def generate(self, shuffle=True):

        while True:
            if shuffle:
                self.img_file_path, self.labels = sklearn.utils.shuffle(self.img_file_path, self.labels)

            for idx in range(0, len(self.img_file_path)):
                anno = self.ssd_encoder.encode_annotation(self.labels[idx])
                yield self.img_file_path[idx], anno

    def sample_count(self):
        return len(self.img_file_path)
