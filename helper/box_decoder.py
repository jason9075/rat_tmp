import tensorflow as tf
import numpy as np

from helper.box_utils import generate_boxes


class DecodePrediction(tf.keras.layers.Layer):
    def __init__(self,
                 img_size,
                 predictor_sizes,
                 aspect_ratios,
                 scales,
                 confidence_thresh=0.01,
                 iou_threshold=0.45,
                 top_k=100,
                 nms_max_output_size=200,
                 **kwargs):
        # We need these members for TensorFlow.
        self.tf_img_height = tf.constant(img_size[0], dtype=tf.float32, name='img_height')
        self.tf_img_width = tf.constant(img_size[1], dtype=tf.float32, name='img_width')
        self.tf_confidence_thresh = tf.constant(confidence_thresh, name='confidence_thresh')
        self.tf_iou_threshold = tf.constant(iou_threshold, name='iou_threshold')
        self.tf_top_k = tf.constant(top_k, name='top_k')
        self.tf_nms_max_output_size = tf.constant(nms_max_output_size, name='nms_max_output_size')
        box_tensor = generate_boxes(img_size, predictor_sizes, aspect_ratios, scales)
        self.box_tensor = tf.constant(box_tensor, name='box_tensor', dtype=tf.float32)

        super(DecodePrediction, self).__init__(**kwargs)


    def build(self, input_shape):
        self.input_spec = [tf.keras.layers.InputSpec(shape=input_shape)]
        super(DecodePrediction, self).build(input_shape)

    def call(self, y_pred, mask=None):
        #####################################################################################
        # 1. Convert the box coordinates from predicted anchor box offsets to predicted
        #    absolute coordinates
        #####################################################################################

        # Extract the predicted class IDs as the indices of the highest confidence values.
        class_ids = tf.expand_dims(tf.cast(tf.argmax(y_pred[..., :-4], axis=-1), tf.float32), axis=-1)
        # Extract the confidences of the maximal classes.
        confidences = tf.math.reduce_max(y_pred[..., :-4], axis=-1, keepdims=True)

        # Convert anchor box offsets to image offsets.
        xy_var = tf.constant(0.1, dtype=tf.float32)
        wh_var = tf.constant(0.2, dtype=tf.float32)

        # cx = cx_pred * cx_variance * w_anchor + cx_anchor
        cx = y_pred[..., -4] * xy_var * self.box_tensor[..., -2] + self.box_tensor[..., -4]
        # cy = cy_pred * cy_variance * h_anchor + cy_anchor
        cy = y_pred[..., -3] * xy_var * self.box_tensor[..., -1] + self.box_tensor[..., -3]
        # w = exp(w_pred * variance_w) * w_anchor
        w = tf.exp(y_pred[..., -2] * wh_var) * self.box_tensor[..., -2]
        # h = exp(h_pred * variance_h) * h_anchor
        h = tf.exp(y_pred[..., -1] * wh_var) * self.box_tensor[..., -1]

        # Convert 'centroids' to 'corners'.
        xmin = cx - 0.5 * w
        ymin = cy - 0.5 * h
        xmax = cx + 0.5 * w
        ymax = cy + 0.5 * h

        xmin = tf.expand_dims(tf.clip_by_value(xmin, clip_value_min=0.0, clip_value_max=1.0), axis=-1)
        ymin = tf.expand_dims(tf.clip_by_value(ymin, clip_value_min=0.0, clip_value_max=1.0), axis=-1)
        xmax = tf.expand_dims(tf.clip_by_value(xmax, clip_value_min=0.0, clip_value_max=1.0), axis=-1)
        ymax = tf.expand_dims(tf.clip_by_value(ymax, clip_value_min=0.0, clip_value_max=1.0), axis=-1)

        y_pred = tf.concat(values=[class_ids, confidences, xmin, ymin, xmax, ymax], axis=-1)

        #####################################################################################
        # 2. Perform confidence thresholding, non-maximum suppression, and top-k filtering.
        #####################################################################################

        # Create a function that filters the predictions for the given batch item. Specifically, it performs:
        # - confidence thresholding
        # - non-maximum suppression (NMS)
        # - top-k filtering
        def filter_predictions(batch_item):
            # Keep only the non-background boxes.
            positive_boxes = tf.not_equal(batch_item[..., 0], 0.0)
            predictions = tf.boolean_mask(tensor=batch_item,
                                          mask=positive_boxes)

            def perform_confidence_thresholding():
                # Apply confidence thresholding.
                threshold_met = predictions[:, 1] > self.tf_confidence_thresh
                return tf.boolean_mask(tensor=predictions,
                                       mask=threshold_met)

            def no_positive_boxes():
                return tf.constant(value=0.0, shape=(1, 6))

            # If there are any positive predictions, perform confidence thresholding.
            predictions_conf_thresh = tf.cond(tf.equal(tf.size(predictions), 0), no_positive_boxes,
                                              perform_confidence_thresholding)

            def perform_nms():
                scores = predictions_conf_thresh[..., 1]

                # `tf.image.non_max_suppression()` needs the box coordinates in the format `(ymin, xmin, ymax, xmax)`.
                xmin = tf.expand_dims(predictions_conf_thresh[..., -4], axis=-1)
                ymin = tf.expand_dims(predictions_conf_thresh[..., -3], axis=-1)
                xmax = tf.expand_dims(predictions_conf_thresh[..., -2], axis=-1)
                ymax = tf.expand_dims(predictions_conf_thresh[..., -1], axis=-1)
                boxes = tf.concat(values=[ymin, xmin, ymax, xmax], axis=-1)

                maxima_indices = tf.image.non_max_suppression(boxes=boxes,
                                                              scores=scores,
                                                              max_output_size=self.tf_nms_max_output_size,
                                                              iou_threshold=self.tf_iou_threshold,
                                                              name='non_maximum_suppresion')
                maxima = tf.gather(params=predictions_conf_thresh,
                                   indices=maxima_indices,
                                   axis=0)
                return maxima

            def no_confident_predictions():
                return tf.constant(value=0.0, shape=(1, 6))

            # If any boxes made the threshold, perform NMS.
            predictions_nms = tf.cond(tf.equal(tf.size(predictions_conf_thresh), 0), no_confident_predictions,
                                      perform_nms)

            # Perform top-k filtering for this batch item or pad it in case there are
            # fewer than `self.top_k` boxes left at this point. Either way, produce a
            # tensor of length `self.top_k`. By the time we return the final results tensor
            # for the whole batch, all batch items must have the same number of predicted
            # boxes so that the tensor dimensions are homogenous. If fewer than `self.top_k`
            # predictions are left after the filtering process above, we pad the missing
            # predictions with zeros as dummy entries.
            def top_k():
                return tf.gather(params=predictions_nms,
                                 indices=tf.nn.top_k(predictions_nms[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            def pad_and_top_k():
                padded_predictions = tf.pad(tensor=predictions_nms,
                                            paddings=[[0, self.tf_top_k - tf.shape(predictions_nms)[0]], [0, 0]],
                                            mode='CONSTANT',
                                            constant_values=0.0)
                return tf.gather(params=padded_predictions,
                                 indices=tf.nn.top_k(padded_predictions[:, 1], k=self.tf_top_k, sorted=True).indices,
                                 axis=0)

            top_k_boxes = tf.cond(tf.greater_equal(tf.shape(predictions_nms)[0], self.tf_top_k), top_k, pad_and_top_k)

            return top_k_boxes

        # Iterate `filter_predictions()` over all batch items.
        output_tensor = tf.map_fn(fn=lambda x: filter_predictions(x),
                                  elems=y_pred,
                                  dtype=None,
                                  parallel_iterations=128,
                                  back_prop=False,
                                  swap_memory=False,
                                  infer_shape=True,
                                  name='loop_over_batch')

        return output_tensor

    def compute_output_shape(self, input_shape):
        batch_size, n_boxes, last_axis = input_shape
        return batch_size, self.tf_top_k, 6  # Last axis: (class_ID, confidence, 4 box coordinates)
