import numpy as np

from helper.box_utils import corners2centroids, iou, match_bipartite_greedy, match_multi

BACKGROUND_ID = 0


class SSDEncoder:

    def __init__(self, input_size, aspect_ratios, scales, n_classes, predictor_sizes):
        self.img_height = input_size[0]
        self.img_width = input_size[1]
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.boxes_list = []

        for i in range(len(self.predictor_sizes)):
            boxes = self._generate_anchor_boxes_for_layer(
                feature_map_size=self.predictor_sizes[i],
                aspect_ratios=self.aspect_ratios[i],
                this_scale=self.scales[i],
                next_scale=self.scales[i + 1])
            self.boxes_list.append(boxes)

    def boxes_count(self):
        count = 0
        for boxes in self.boxes_list:
            w, h, n, _ = boxes.shape
            count += w * h * n

        return count

    def encode_annotation(self, anno):
        anno = np.array(anno)
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        y_encoded = self._generate_encoding_template()  # shape=(#num_box, #classes + #loc + #anchor_loc)
        y_encoded[:, BACKGROUND_ID] = 1
        class_vectors = np.eye(self.n_classes)

        if len(anno) != 0:
            self.matching_gt_to_anchor(anno, class_id, class_vectors, xmax, xmin, y_encoded, ymax, ymin)

        xy_var = 0.1
        wh_var = 0.2

        # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        y_encoded[:, [-8, -7]] -= y_encoded[:, [-4, -3]]
        # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_encoded[:, [-8, -7]] /= y_encoded[:, [-2, -1]] * [[xy_var, xy_var]]
        # w(gt) / w(anchor), h(gt) / h(anchor)
        y_encoded[:, [-6, -5]] /= y_encoded[:, [-2, -1]]
        # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        y_encoded[:, [-6, -5]] = np.log(y_encoded[:, [-6, -5]]) / [[wh_var, wh_var]]

        return y_encoded

    def matching_gt_to_anchor(self, anno, class_id, class_vectors, xmax, xmin, y_encoded, ymax, ymin):
        labels = anno.astype(np.float)
        # 因訓練圖片長寬不一，所以都用比例表示
        labels[:, [ymin, ymax]] /= self.img_height
        labels[:, [xmin, xmax]] /= self.img_width
        similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[:, -8:-4])
        labels = corners2centroids(labels, start_index=xmin)
        classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]
        labels_one_hot = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1)
        # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
        #        This ensures that each ground truth box will have at least one good match.
        # For each ground truth box, get the anchor box to match with it.
        bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)
        # Write the ground truth data to the matched anchor boxes.
        y_encoded[bipartite_matches, :-4] = labels_one_hot
        # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
        similarities[:, bipartite_matches] = 0
        # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
        #         ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
        #         such ground truth box.
        pos_iou_threshold = 0.5
        # Get all matches that satisfy the IoU threshold.
        matches = match_multi(weight_matrix=similarities, threshold=pos_iou_threshold)
        # Write the ground truth data to the matched anchor boxes.
        y_encoded[matches[1], :-4] = labels_one_hot[matches[0]]
        # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
        similarities[:, matches[1]] = 0
        # Third: Now after the matching is done, all negative (background) anchor boxes that have
        #        an IoU of `neg_iou_limit` or more with any ground truth box will be set to netral,
        #        i.e. they will no longer be background boxes. These anchors are "too close" to a
        #        ground truth box to be valid background boxes.
        neg_iou_limit = 0.3
        max_background_similarities = np.amax(similarities, axis=0)
        neutral_boxes = np.nonzero(neg_iou_limit <= max_background_similarities)[0]
        y_encoded[neutral_boxes, BACKGROUND_ID] = 0

    def _generate_anchor_boxes_for_layer(self,
                                         feature_map_size,
                                         aspect_ratios,
                                         this_scale,
                                         next_scale):
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in aspect_ratios:
            if ar == 1:
                # Compute the regular anchor box for aspect ratio 1.
                box_height = box_width = this_scale * size
                wh_list.append((box_width, box_height))
                # Compute one slightly larger version using the geometric mean of this scale value and the next.
                box_height = box_width = np.sqrt(this_scale * next_scale) * size
                wh_list.append((box_width, box_height))
            else:
                box_width = this_scale * size * np.sqrt(ar)
                box_height = this_scale * size / np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)
        n_boxes = len(wh_list)

        step_height = self.img_height / feature_map_size[0]
        step_width = self.img_width / feature_map_size[1]
        offset_height = 0.5
        offset_width = 0.5

        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height,
                         feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width,
                         feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)  # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)  # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # normalize the coordinates to be within [0,1]
        boxes_tensor[:, :, :, [0, 2]] /= self.img_width
        boxes_tensor[:, :, :, [1, 3]] /= self.img_height

        return boxes_tensor

    def _generate_encoding_template(self):
        boxes_batch = []
        for boxes in self.boxes_list:
            boxes = np.reshape(boxes, (-1, 4))
            boxes_batch.append(boxes)

        boxes_tensor = np.concatenate(boxes_batch, axis=0)

        classes_tensor = np.zeros((boxes_tensor.shape[0], self.n_classes))

        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor), axis=1)

        return y_encoding_template
