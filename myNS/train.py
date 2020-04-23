# coding: utf-8
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import os

slim = tf.contrib.slim

# 参数设置
loss_model = "vgg_16"
loss_modle_file="../pretrained/vgg_16.ckpt"
style_image = "img/wave.jpg"
image_size = 256
exclusions = "vgg_16/fc"
model_path = "models"
naming: "feathers"  # 训练的风格图叫啥
# list
style_layers = ["vgg_16/conv1/conv1_2",
                "vgg_16/conv2/conv2_2",
                "vgg_16/conv3/conv3_3",
                "vgg_16/conv4/conv4_3"]


def get_image(path, height, width, preprocess_fn):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)

# 从数据集中取得图片
def image(batch_size, height, width, path, preprocess_fn, epochs=2, shuffle=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)

    png = filenames[0].lower().endswith('png')  # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)

    processed_image = preprocess_fn(image, height, width)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)



def _get_init_fn():
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)



def gram(layer):  # gram矩阵实现  
    shape = tf.shape(layer)
    sh    = layer.shape 
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    #print num_images,sh,width,height,num_filters
    # stack 矩阵拼接
    # print tf.stack([num_images, -1, num_filters])
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))  #把layer的w h合并
    #print "reshp" , filters.shape
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


# 读入风格图片 并处理 输出制定中间层的数据
def get_style_features():
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            loss_model,num_classes=1,is_training=False
        )
        image_preprocessing_fn,image_unprocessing_fn = preprocessing_factory.get_preprocessing(
            loss_model,False
        )
    size = image_size
    img_bytes = tf.read_file(style_image) # 风格图数据
    image = tf.image.decode_jpeg(img_bytes)
    # 256 256 3 -> 0 256 256 3 在最前面增加一个维度
    images = tf.expand_dims(image_preprocessing_fn(image,size,size),0)

    # 得到风格图像各层网络中的矩阵
    _, endpoints_dict = network_fn(images,spatial_squeeze=False)
    # 测试endpoints_dict
    #for k,v in endpoints_dict.items():
    #    print k,"---" ,v

    features = []
    for layer in style_layers:
        feature = endpoints_dict[layer]
        #print gram(features)
        feature = tf.squeeze(gram(feature),[0])  # 变成2维
        # print features
        features.append(feature) # 4个n*n的
    # ？ 初始化网络 
    with tf.Session() as sess:
        init_func = _get_init_fn()
        init_func(sess)
    
    if os.path.exists('generated') is False:
                os.makedirs('generated')
    save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
    
    with open(save_file, 'wb') as f:
        target_image = image_unprocessing_fn(images[0, :])
        value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
        f.write(sess.run(value))
        tf.logging.info('Target style pattern is saved to: %s.' % save_file)
    return sess.run(features)

# 读入coco数据集

def main():
    style_features_t = get_style_features()
    # ckpt
    training_path = os.path.join(model_path,naming)
    if not(os.path.exists(training_path)):
        os.makedirs(training_path)
    # 构建计算损失的网络
    with tf.Graph().as_default():
        with tf.Session() as sess:
            net_work_fn = nets_factory.get_network_fn(
                loss_model,num_classes=1,is_training=False
            )
            image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
                    loss_model,
                    is_training=False)
            image()
            
        

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()