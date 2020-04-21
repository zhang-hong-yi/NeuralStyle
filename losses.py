# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import utils
import os

slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)   
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))  #把layer的w h合并
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


    """
        处理风格图像：
        一。将较短的边调整为FLAGS.image大小
        2。应用中心裁剪
    """
def get_style_features(FLAGS):
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)
        image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            FLAGS.loss_model,
            is_training=False)

        # 读取风格图像
        size = FLAGS.image_size
        img_bytes = tf.read_file(FLAGS.style_image)
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes)
        else:
            image = tf.image.decode_jpeg(img_bytes)
        

        # Add the batch dimension  添加维度
        images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        

        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []
        for layer in FLAGS.style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])  # 删除批次维度
            features.append(feature)    #获得需要的层的风格图像的数据

        with tf.Session() as sess:
            # 取得损失网络的参数
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)

            # 保存处理过的风格图片
            if os.path.exists('generated') is False:
                os.makedirs('generated')
           
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
           
            with open(save_file, 'wb') as f:
                target_image = image_unprocessing_fn(images[0, :])
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            # #获得需要的层的风格图像的数据
            return sess.run(features)

# 定义风格损失
# 原始图片和 生成图片的所有层合并的数组
 #style_features_t 风格图片的层数据
# style_layers 定义使用哪些层计算风格损失
def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {} #tensorboard
    for style_gram, layer in zip(style_features_t, style_layers):
        # 计算生成图片，只计算生成图片generated_images与目标风格
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        #计算损失
        layer_style_loss = tf.nn.l2_loss(gram(generated_images) - style_gram) * 2 / tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary


# endpoints_dict 是损失网络各层的计算结果
# content_layers 是定义使用哪些层的差距计算损失
def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:  #把生成图像和原始图像区分开
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)  
        # 内容损失
        content_loss += tf.nn.l2_loss(generated_images - content_images) * 2 / tf.to_float(size)  
    return content_loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height - 1, -1, -1])) - tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width - 1, -1])) - tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))
    return loss
