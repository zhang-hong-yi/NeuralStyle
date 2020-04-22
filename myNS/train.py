# coding: utf-8
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
# 参数设置
loss_model = "vgg_16"
loss_modle_file="../pretrained/vgg_16.ckpt"
style_image = "img/wave.jpg"
image_size = 256

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

    _, endpoints_dict = network_fn(images,spatial_squeeze=False)
    # print(endpoints_dict.size())
    for k,v in endpoints_dict.items():
        print k,"aaa" ,v

# 读入coco数据集



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    get_style_features()