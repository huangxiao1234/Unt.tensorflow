from __future__ import print_function
import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
from collections import OrderedDict


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 21
IMAGE_SIZE = 320
# 训练模式将上面的mode换成train,可视化就viualize


def inferenceUNET(image, keep_prob,layers=3):
    # 还没考虑dropout 和 带权重进行交叉熵
    features_root = 16
    filter_size = 3
    pool_size = 2
    channels = 3
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    pools = OrderedDict()
    deconv = OrderedDict()
    in_node = image

    for layer in range(0, layers):
        with tf.name_scope("down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root  # 论文第一层的feature_num不是64吗？为啥这里是16
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = utils.weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
            else:
                w1 = utils.weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

            w2 = utils.weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            b1 = utils.bias_variable([features], name="b1")
            b2 = utils.bias_variable([features], name="b2")

            conv1 = utils.conv2d_basic(in_node, w1, b1)
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = utils.conv2d_basic(tmp_h_conv, w2, b2)
            dw_h_convs[layer] = tf.nn.relu(conv2)

            if layer < layers - 1:  # 意味着这里的layers是算到最下面那层的，所以论文的有5层，这里说的layer指的level,每个level固定3层
                pools[layer] = utils.max_pool_2x2(dw_h_convs[layer])
                in_node = pools[layer]

    in_node = dw_h_convs[layers - 1]

    # up layers
    for layer in range(layers - 2, -1, -1):  # [3 2 1 0]
        with tf.name_scope("up_conv_{}".format(str(layer))):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            # 要注意的是，上采样从小变大靠的是反卷积，而下采样从大变小靠的是池化，上采样是没有池化的
            wd = utils.weight_variable([pool_size, pool_size, features // 2, features], stddev,
                                       name="wd")  # 注意了这里反卷积参数位置与卷积的位置在in_channel和out_channel相反
            bd = utils.bias_variable([features // 2], name="bd")
            h_deconv = tf.nn.relu(utils.deconv2d(in_node, wd, pool_size) + bd)
            h_deconv_concat = utils.crop_and_concat(dw_h_convs[layer],
                                                    h_deconv)  # 把前面上采样层的对应进行拼接，具体过程就是裁剪下采样的样本，使尺寸相同，然后拼在后面即可，接的是池化前那一层，而不是统一Level的所有层
            deconv[layer] = h_deconv_concat

            w1 = utils.weight_variable([filter_size, filter_size, features, features // 2], stddev,
                                       name="w1")  # 上面反卷积然后拼接，feature不是变多了吗？为啥这里还用feature.看漏了。。上面反卷积的时候feature变少了一半
            w2 = utils.weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
            b1 = utils.bias_variable([features // 2], name="b1")
            b2 = utils.bias_variable([features // 2], name="b2")

            conv1 = utils.conv2d_basic(h_deconv_concat, w1, b1)
            h_conv = tf.nn.relu(conv1)
            conv2 = utils.conv2d_basic(h_conv, w2, b2)
            in_node = tf.nn.relu(conv2)
            up_h_convs[layer] = in_node

    weight = utils.weight_variable([1, 1, features_root, NUM_OF_CLASSESS], stddev)
    bias = utils.bias_variable([NUM_OF_CLASSESS], name="bias")
    conv = utils.conv2d_basic(in_node, weight, bias)
    output_map = tf.nn.relu(conv)
    up_h_convs["out"] = output_map
    annotation_pred = tf.argmax(output_map, dimension=3, name="prediction")

    return annotation_pred,output_map


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inferenceUNET(image, keep_probability,layers=4)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))# 注意啊，这个和softmax_cross_entropy_with_logits()不一样，这个输入的label还要转one-hot编码的，而softmax_cross_entropy_with_logits()是对于那些已经转好的label。因此这个输入的label是一张单通道的图片，进去后会变成Num_class通道的Onthot矩阵，这样就能交叉熵啦
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:# 这个返回的是最新的model路径，而all_model_checkpoint_path是全部的
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 50 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(10)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred) # reshape一下

        for itr in range(10):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
