import tensorflow as tf
import tensorflow.contrib.slim as slim
from contextlib import ExitStack

class vgg16(object):
    def __init__(self, batch_size=1,image=None):
        self.batch_size=1
        self.image=image
        self.layers={}
    def build_network(self, sess, is_training=True):
        net = self.build_head(self.image,is_training)
        blocklength=self.build_BLnet(net)
        output=self.build_discrimitor(net)
        self.layers['output']=output

    def build_head(self,net,is_training=True):
        with tf.variable_scope('vgg_16'):
            with ExitStack() as stack:
                stack.enter_context(
                    slim.arg_scope(
                        [ slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(2.5e-5),
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        normalizer_fn=slim.batch_norm,
                    ))
                net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')

                # Layer 2
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')

                # Layer 3
                net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], trainable=is_training, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')

                # Layer 4
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], trainable=is_training, scope='conv4')

                # Append network as head layer
                self.layers['head'] = net

    def build_BLnet(self,net):
        with tf.variable_scope('BLnet'):
            with ExitStack() as stack:
                stack.enter_context(slim.arg_scope(
                                                    [ slim.fully_connected],
                                                    activation_fn=tf.nn.relu,
                                                    weights_regularizer=slim.l2_regularizer(2.5e-5),
                                                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                    normalizer_fn=slim.batch_norm))
                net=slim.fully_connected(net,256,scope='fc1')
                net = slim.fully_connected(net, 64, scope='fc2')
                net = slim.dropout(net, keep_prob=0.5, is_training=True, scope='dropout3')
                net = slim.fully_connected(net, 32, scope='fc4')
                self.layers['blocklength']=net
                return net

    def build_discrimitor(self,net,blocklenth):

        with tf.variable_scope('discrimitor'):
            with ExitStack() as stack:
                stack.enter_context(
                    slim.arg_scope(
                        [ slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(2.5e-5),
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        normalizer_fn=slim.batch_norm))
                net=slim.conv2d(net,1024,[3,3],padding='same',name='conv1')

                net=self.max_pool(net,2,2,2,1,padding='valid',name='pool2')
                net=slim.conv2d(net,1024,[2,1],padding='same',name='conv3')
                net = slim.conv2d(net, 11, [1, 1], padding='same', name='conv4')
                return net



    def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
                 padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)


