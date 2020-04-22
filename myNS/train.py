# coding: utf-8
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
# 参数设置
loss_model = "vgg_16"
loss_modle_file="../pretrained/vgg_16.ckpt"
style_image = "img/wave.jpg"
image_size = 256

# list
style_layers = ["vgg_16/conv1/conv1_2",
                "vgg_16/conv2/conv2_2",
                "vgg_16/conv3/conv3_3",
                "vgg_16/conv4/conv4_3"]


def gram(layer):  # gram矩阵实现  
    shape = tf.shape(layer)
    sh    = layer.shape 
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    #print num_images,sh,width,height,num_filters
    # stack 矩阵拼接
    print tf.stack([num_images, -1, num_filters])
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
        features.append(feature)
    with tf.Session() as sess:
        init_func = 

# 读入coco数据集



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    get_style_features()