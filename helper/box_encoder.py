import numpy as np

from helper.box_utils import corners2centroids, iou, match_bipartite_greedy, match_multi, generate_boxes

BACKGROUND_ID = 0


class SSDEncoder:

    def __init__(self, input_size, aspect_ratios, scales, n_classes, predictor_sizes):
        self.img_height = input_size[0]
        self.img_width = input_size[1]
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes

        self.boxes_list = generate_boxes(input_size, predictor_sizes, aspect_ratios, scales)

    def boxes_count(self):
        return self.boxes_list.shape[0]

    def encode_annotation(self, anno):
        anno = np.array(anno)
        class_id = 0
        xmin = 1
        ymin = 2
        xmax = 3
        ymax = 4

        y_encoded = self._generate_encoding_template()  # shape=(#num_box, #classes + #loc)
        y_encoded[:, BACKGROUND_ID] = 1
        class_vectors = np.eye(self.n_classes)

        if len(anno) != 0:
            self.matching_gt_to_anchor(anno, class_id, class_vectors, xmax, xmin, y_encoded, ymax, ymin)

        xy_var = 0.1
        wh_var = 0.2

        # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        y_encoded[:, [-4, -3]] -= self.boxes_list[:, [-4, -3]]
        # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_encoded[:, [-4, -3]] /= self.boxes_list[:, [-2, -1]] * xy_var
        # w(gt) / w(anchor), h(gt) / h(anchor)
        y_encoded[:, [-2, -1]] /= self.boxes_list[:, [-2, -1]]
        # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        y_encoded[:, [-2, -1]] = np.log(y_encoded[:, [-2, -1]]) / wh_var

        return y_encoded

    @staticmethod
    def matching_gt_to_anchor(anno, class_id, class_vectors, xmax, xmin, y_encoded, ymax, ymin):
        labels = anno.astype(np.float)
        similarities = iou(labels[:, [xmin, ymin, xmax, ymax]], y_encoded[:, -4:])
        labels = corners2centroids(labels, start_index=xmin)
        classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)]
        labels_updates = np.concatenate([classes_one_hot, labels[:, [xmin, ymin, xmax, ymax]]], axis=-1)
        # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest IoU.
        #        This ensures that each ground truth box will have at least one good match.
        # For each ground truth box, get the anchor box to match with it.
        bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)
        # Write the ground truth data to the matched anchor boxes.
        y_encoded[bipartite_matches, :] = labels_updates
        # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
        similarities[:, bipartite_matches] = 0
        # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
        #         ground truth box with an IoU of at least `pos_iou_threshold`, or not matched if there is no
        #         such ground truth box.
        pos_iou_threshold = 0.5
        # Get all matches that satisfy the IoU threshold.
        matches = match_multi(weight_matrix=similarities, threshold=pos_iou_threshold)
        # Write the ground truth data to the matched anchor boxes.
        y_encoded[matches[1], :] = labels_updates[matches[0]]
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

    def _generate_encoding_template(self):
        classes_tensor = np.zeros((self.boxes_list.shape[0], self.n_classes))

        y_encoding_template = np.concatenate((classes_tensor, self.boxes_list), axis=1)

        return y_encoding_template
