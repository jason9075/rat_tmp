import tensorflow as tf


def mobilenet_v2(input_shape, n_classes, aspect_ratios, model_type):
    boxes_per_layer = []
    for ar in aspect_ratios:
        if 1 in ar:
            boxes_per_layer.append(len(ar) + 1)
        else:
            boxes_per_layer.append(len(ar))

    layer_depths = [-1, -1, 512, 256, 256, 128]

    full_mobile_net_v2 = tf.keras.applications.MobileNetV2(include_top=False,
                                                           input_shape=(input_shape[0], input_shape[1], 3))

    feature_map_1 = full_mobile_net_v2.get_layer(name='block_13_expand_relu').output
    feature_map_2 = full_mobile_net_v2.get_layer(name='out_relu').output
    feature_map_3 = gen_fmap('BoxPredictor3', feature_map_2, layer_depths[2])
    feature_map_4 = gen_fmap('BoxPredictor4', feature_map_3, layer_depths[3])
    feature_map_5 = gen_fmap('BoxPredictor5', feature_map_4, layer_depths[4])
    feature_map_6 = gen_fmap('BoxPredictor6', feature_map_5, layer_depths[5])

    conf_1, loc_1 = gen_conf_and_loc(feature_map_1, boxes_per_layer[0], n_classes, 'BoxPredictor1')
    conf_2, loc_2 = gen_conf_and_loc(feature_map_2, boxes_per_layer[1], n_classes, 'BoxPredictor2')
    conf_3, loc_3 = gen_conf_and_loc(feature_map_3, boxes_per_layer[2], n_classes, 'BoxPredictor3')
    conf_4, loc_4 = gen_conf_and_loc(feature_map_4, boxes_per_layer[3], n_classes, 'BoxPredictor4')
    conf_5, loc_5 = gen_conf_and_loc(feature_map_5, boxes_per_layer[4], n_classes, 'BoxPredictor5')
    conf_6, loc_6 = gen_conf_and_loc(feature_map_6, boxes_per_layer[5], n_classes, 'BoxPredictor6')

    confidence = tf.keras.layers.Concatenate(axis=1, name='confidence')(
        [conf_1, conf_2, conf_3, conf_4, conf_5, conf_6])
    confidence = tf.keras.layers.Activation('softmax', name='confidence_softmax')(confidence)

    locations = tf.keras.layers.Concatenate(axis=1, name='locations')([loc_1, loc_2, loc_3, loc_4, loc_5, loc_6])

    predictions = tf.keras.layers.Concatenate(axis=2, name='predictions')([confidence, locations, locations])

    model = tf.keras.Model(inputs=full_mobile_net_v2.inputs, outputs=[predictions])
    # if model_type == 'train':
    #     model = tf.keras.Model(inputs=full_mobile_net_v2.inputs, outputs=[confidence, locations])
    # else:
    #     model = tf.keras.Model(inputs=full_mobile_net_v2.inputs, outputs=[confidence, locations])

    return model


def gen_conf_and_loc(feature_map, n_boxes, n_classes, base_name):
    conf = tf.keras.layers.Conv2D(n_boxes * n_classes,
                                  (3, 3),
                                  padding='SAME',
                                  kernel_initializer='he_normal',
                                  name=base_name + '_conf_conv')(feature_map)
    conf = tf.keras.layers.Reshape((-1, n_classes), name=base_name + '_conf')(conf)

    loc = tf.keras.layers.Conv2D(n_boxes * 4,
                                 (3, 3), padding='SAME',
                                 kernel_initializer='he_normal',
                                 name=base_name + '_loc_conv')(feature_map)
    loc = tf.keras.layers.Reshape((-1, 4), name=base_name + '_loc')(loc)

    return conf, loc


def gen_fmap(base_name, net, layer_depth):
    layer_name = '{}_1_Conv2d_1x1_s1'.format(
        base_name)
    net = tf.keras.layers.Conv2D(int(layer_depth / 2),
                                 (1, 1),
                                 padding='SAME',
                                 kernel_initializer='he_normal',
                                 strides=1,
                                 name=layer_name + '_conv')(net)
    net = tf.keras.layers.BatchNormalization(name=layer_name + '_bn')(net)
    net = tf.keras.layers.Lambda(tf.nn.relu6, name=layer_name + '_relu6')(net)

    layer_name = '{}_2_Conv2d_3x3_s2'.format(base_name)
    net = tf.keras.layers.DepthwiseConv2D(
        (3, 3),
        padding='SAME',
        strides=2,
        name=layer_name + '_depthwise_conv')(net)
    net = tf.keras.layers.BatchNormalization(name=layer_name + '_depthwise_bn')(net)
    net = tf.keras.layers.Lambda(tf.nn.relu6, name=layer_name + '_depthwise_relu6')(net)

    net = tf.keras.layers.Conv2D(layer_depth,
                                 (1, 1),
                                 padding='SAME',
                                 kernel_initializer='he_normal',
                                 strides=1,
                                 name=layer_name + '_conv')(net)
    net = tf.keras.layers.BatchNormalization(name=layer_name + '_bn')(net)
    net = tf.keras.layers.Lambda(tf.nn.relu6, name=layer_name + '_relu6')(net)

    return net
