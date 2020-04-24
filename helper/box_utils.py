import numpy as np


def corners2centroids(labels, start_index=1):
    ind = start_index
    outputs = np.copy(labels).astype(np.float)

    outputs[..., ind] = (labels[..., ind] + labels[..., ind + 2]) / 2.0  # Set cx
    outputs[..., ind + 1] = (labels[..., ind + 1] + labels[..., ind + 3]) / 2.0  # Set cy
    outputs[..., ind + 2] = labels[..., ind + 2] - labels[..., ind]  # Set w
    outputs[..., ind + 3] = labels[..., ind + 3] - labels[..., ind + 1]  # Set h

    return outputs


def centroids2corners(labels, start_index=1):
    ind = start_index
    outputs = np.copy(labels).astype(np.float)

    outputs[..., ind] = labels[..., ind] - labels[..., ind + 2] / 2.0  # Set xmin
    outputs[..., ind + 1] = labels[..., ind + 1] - labels[..., ind + 3] / 2.0  # Set ymin
    outputs[..., ind + 2] = labels[..., ind] + labels[..., ind + 2] / 2.0  # Set xmax
    outputs[..., ind + 3] = labels[..., ind + 1] + labels[..., ind + 3] / 2.0  # Set ymax

    return outputs


def iou(gt_corner, anchor_centroid):
    anchor_corner = centroids2corners(anchor_centroid, start_index=0)

    intersection_areas = intersection_area_(gt_corner, anchor_corner)

    m = gt_corner.shape[0]  # The number of boxes in `gt`
    n = anchor_corner.shape[0]  # The number of boxes in `anchor`

    # corner coordinate
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    gt_areas = np.tile(
        np.expand_dims((gt_corner[:, xmax] - gt_corner[:, xmin]) * (gt_corner[:, ymax] - gt_corner[:, ymin]), axis=1),
        reps=(1, n))
    anchor_areas = np.tile(
        np.expand_dims((anchor_corner[:, xmax] - anchor_corner[:, xmin]) * (
                anchor_corner[:, ymax] - anchor_corner[:, ymin]), axis=0),
        reps=(m, 1))

    union_areas = gt_areas + anchor_areas - intersection_areas

    return intersection_areas / union_areas


def intersection_area_(boxes1, boxes2):
    m = boxes1.shape[0]
    n = boxes2.shape[0]

    # corner coordinate
    xmin = 0
    ymin = 1
    xmax = 2
    ymax = 3

    min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [xmin, ymin]], axis=1), reps=(1, n, 1)),
                        np.tile(np.expand_dims(boxes2[:, [xmin, ymin]], axis=0), reps=(m, 1, 1)))

    max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [xmax, ymax]], axis=1), reps=(1, n, 1)),
                        np.tile(np.expand_dims(boxes2[:, [xmax, ymax]], axis=0), reps=(m, 1, 1)))

    side_lengths = np.maximum(0, max_xy - min_xy)

    return side_lengths[:, :, 0] * side_lengths[:, :, 1]


def match_bipartite_greedy(weight_matrix):
    weight_matrix = np.copy(weight_matrix)  # We'll modify this array.
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes))  # Only relevant for fancy-indexing below.

    # This 1D array will contain for each ground truth box the index of
    # the matched anchor box.
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)

    # In each iteration of the loop below, exactly one ground truth box
    # will be matched to one anchor box.
    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1)  # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps)  # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index  # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:, anchor_index] = 0

    return matches


def match_multi(weight_matrix, threshold):
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))  # Only relevant for fancy-indexing below.

    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0)  # Array of shape (weight_matrix.shape[1],)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]  # Array of shape (weight_matrix.shape[1],)

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met


def generate_boxes(img_hw, predictor_sizes, aspect_ratios, scales, offset_height=0.5, offset_width=0.5):
    size = min(img_hw)

    boxes_tensor = []
    for idx, (feature_map_height, feature_map_width) in enumerate(predictor_sizes):
        step_height = img_hw[0] / feature_map_height
        step_width = img_hw[1] / feature_map_width

        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height,
                         feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width,
                         feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1)
        cy_grid = np.expand_dims(cy_grid, -1)

        wh_list = []
        for ar in aspect_ratios[idx]:
            if ar == 1:
                box_height = box_width = scales[idx] * size
                wh_list.append((box_width, box_height))
                box_height = box_width = np.sqrt(scales[idx] * scales[idx + 1]) * size
                wh_list.append((box_width, box_height))
            else:
                box_height = scales[idx] * size / np.sqrt(ar)
                box_width = scales[idx] * size * np.sqrt(ar)
                wh_list.append((box_width, box_height))
        wh_list = np.array(wh_list)

        n_boxes = wh_list.shape[0]
        boxes_tensor_per_layer = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))
        boxes_tensor_per_layer[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor_per_layer[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor_per_layer[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor_per_layer[:, :, :, 3] = wh_list[:, 1]  # Set h

        boxes_tensor_per_layer[:, :, :, [1, 3]] /= img_hw[0]  # divide image height
        boxes_tensor_per_layer[:, :, :, [0, 2]] /= img_hw[1]  # divide image weight

        boxes_tensor.append(np.reshape(boxes_tensor_per_layer, (-1, 4)))

    boxes_tensor = np.concatenate(boxes_tensor, axis=0)

    return boxes_tensor
