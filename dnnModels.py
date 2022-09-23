#
# Copyright (c) IEE, University of Luxembourg 2019-2020.
# Created by Fabrizio Pastore, fabrizio.pastore@uni.lu, SNT, 2019.
# Created by Jun WANG, jun.wang@iee.lu, IEE, 2019.
# Modified by Hazem FAHMY, hazem.fahmy@uni.lu, SNT, 2019.
#

#from imports import model_zoo, Parameter, F, nn, torch, tf
from imports import model_zoo, Parameter, F, nn, torch, tf
sessG = None
# tf.disable_v2_behavior()

print(tf.__version__)
tf.compat.v1.disable_eager_execution()
__all__ = ['AlexNet', 'alexnet']

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}


class Linear(nn.Linear):
    def __init__(self, inF, outF):
        super(nn.Linear, self).__init__()
        self.in_features = inF
        self.out_features = outF
        self.weight = Parameter(torch.Tensor(outF, inF))
        bias = True
        if bias:
            self.bias = Parameter(torch.Tensor(outF))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # def forward(self, x):
    #    self.AN = x
    #    return x
    def relprop(self, R):
        V = torch.clamp(self.weight, min=0)
        Z = torch.mm(self.X, torch.transpose(V, 0, 1)) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        R = self.X * C
        self.HM = R
        self.AN = self.Y
        return R


class Dropout(nn.Dropout):
    # def forward(self, x):
    #    self.AN = x
    #    return x
    def relprop(self, R):
        self.HM = R
        return R


class ReLU(nn.ReLU):
    # def forward(self, x):
    #    self.AN = x
    #    return x
    def relprop(self, R):
        self.HM = R
        return R


class Reshape_Alex(nn.Module):
    def __init__(self):
        super(Reshape_Alex, self).__init__()

    def forward(self, x):
        x = x.view(-1, 256 * 7 * 7)
        # self.AN = x
        return x

    def relprop(self, R):
        self.HM = R
        return R.view(-1, 256, 7, 7)


class Conv2d(nn.Conv2d):

    def gradprop(self, DY):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, self.weight, stride=self.stride,
                                  padding=self.padding, output_padding=output_padding)

    # def forward(self, x):
    #    x = self.Y
    #    return x
    def test(self):
        print("test")
        return

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        self.HM = R
        return R


class ConvTranspose2d(nn.ConvTranspose2d):
    def gradprop(self, DY):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv2d(DY, self.weight, stride=self.stride,
                        padding=self.padding)

    # def forward(self, x):
    #    x = self.Y
    #    return x

    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        self.AN = self.Y
        self.HM = R
        return R


class MaxPool2d(nn.MaxPool2d):
    def gradprop(self, DY):
        DX = self.X * 0
        temp, indices = F.max_pool2d(self.X, self.kernel_size, self.stride,
                                     self.padding, self.dilation, self.ceil_mode, True)
        DX = F.max_unpool2d(DY, indices, self.kernel_size, self.stride, self.padding)
        return DX

    # def forward(self, x):
    #    self.AN = x
    #    return x
    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        self.HM = R
        return R

class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            Dropout(),
            # Linear(256 * 3 * 3, 4096),
            Linear(256 * 7 * 7, 4096),

            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        self.oldSize = x.size(0)
        # print(x.size())
        # x = x.view(x.size(0), 256 * 3 * 3)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

    def modifyToTrain(self):
        # code to generate simple extension
        self.classifier = nn.Sequential(
            Dropout(),
            # Linear(256 * 3 * 3, 4096),
            Linear(256 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            # This was to compute the size directly
            # Linear(4096, 1),
            Linear(4096, 2),
        )
        for param in self.features.parameters():
            param.requires_grad = False
        # for param in self.classifier.parameters():
        #    param.requires_grad = False
        # self.classifier = torch.nn.Sequential(self.classifier, ReLU(inplace=True), torch.nn.Linear(8, 2))

    def relprop(self, R):
        for l in range(len(self.classifier), 0, -1):
            R = self.classifier[l - 1].relprop(R)
        self.HM = R
        # R = R.view(256, 3, -1)
        R = R.view(256, 7, -1)
        for l in range(len(self.features), 0, -1):
            R = self.features[l - 1].relprop(R)
        return R

    def modifyToPredict(self):
        # removed = list(self.classifier.children())[:-1]
        # self.classifier = torch.nn.Sequential(*removed)
        # self.classifier = torch.nn.Sequential(self.classifier, torch.nn.Linear(4096, 2))

        # code to generate ".extendedLinearLinear.pth"
        # self.classifier = nn.Sequential(
        #    Linear(256 * 7 * 7, 4096),
        #    ReLU(inplace=True),
        #    Linear(4096, 2)
        # )

        # code to generate simple extension
        self.classifier = nn.Sequential(
            Dropout(),
            Linear(256 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, 29),
        )

        # code to generate ".extendedAddSoftmax.pth"
        # self.classifier = nn.Sequential(
        #    Dropout(),
        #    Linear(256 * 7 * 7, 4096),
        #    ReLU(inplace=True),
        #    Dropout(),
        #    Linear(4096, 4096),
        #    ReLU(inplace=True),
        #    Linear(4096, 2),
        #    nn.Softmax()
        # )

        for param in self.features.parameters():
            param.requires_grad = False

        # for param in self.classifier.parameters():
        #    param.requires_grad = False

        # self.classifier = torch.nn.Sequential(self.classifier, ReLU(inplace=True), torch.nn.Linear(8, 2))

    def modifyToPupil(self):
        # code to generate simple extension
        self.classifier = nn.Sequential(
            Dropout(),
            Linear(256 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            # This was to compute the size directly
            # Linear(4096, 1),
            Linear(4096, 3),
        )

        for param in self.features.parameters():
            param.requires_grad = False

        # for param in self.classifier.parameters():
        #    param.requires_grad = False

        # self.classifier = torch.nn.Sequential(self.classifier, ReLU(inplace=True), torch.nn.Linear(8, 2))


class AlexNetIEE(nn.Module):

    def __init__(self, num_classes):
        super(AlexNetIEE, self).__init__()
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            Dropout(),
            Linear(256 * 3 * 3, 4096),
            # Linear(256 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes),
        )

    def modifyToTrain(self):
        # code to generate simple extension
        self.classifier = nn.Sequential(
            Dropout(),
            Linear(256 * 3 * 3, 4096),
            # Linear(256 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            # This was to compute the size directly
            # Linear(4096, 1),
            Linear(4096, 2),
        )
        for param in self.features.parameters():
            param.requires_grad = False
        # for param in self.classifier.parameters():
        #    param.requires_grad = False
        # self.classifier = torch.nn.Sequential(self.classifier, ReLU(inplace=True), torch.nn.Linear(8, 2))

    def forward(self, x):
        x = self.features(x)
        self.oldSize = x.size(0)
        # print(x.size())
        x = x.view(x.size(0), 256 * 3 * 3)
        # x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

    def relprop(self, R):
        for l in range(len(self.classifier), 0, -1):
            R = self.classifier[l - 1].relprop(R)
            # print(R.size())
        self.HM = R
        R = R.view(256, 3, -1)
        # R = R.view(256, 7, -1)
        for l in range(len(self.features), 0, -1):
            R = self.features[l - 1].relprop(R)
            # print(R.size())
        return R


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


class AlexNetPredict(AlexNet):
    def __init__(self, num_classes=29):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=5, padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=3, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )

        ##code for simple extension
        self.classifier = nn.Sequential(
            Dropout(),
            Linear(256 * 7 * 7, 4096),
            ReLU(inplace=True),
            Dropout(),
            Linear(4096, 4096),
            ReLU(inplace=True),
            Linear(4096, num_classes),
        )

        ##code for .extendedLinearLinear.pth
        # self.classifier = nn.Sequential(
        #    Linear(256 * 7 * 7, 4096),
        #    ReLU(inplace=True),
        #    Linear(4096, 2),
        # )

        # code to generate "extendedAddSoftmax.pth"
        # self.classifier = nn.Sequential(
        #    Dropout(),
        #    Linear(256 * 7 * 7, 4096),
        #    ReLU(inplace=True),
        #    Dropout(),
        #    Linear(4096, 4096),
        #    ReLU(inplace=True),
        #    Linear(4096, 2),
        #    nn.Softmax()
        # )


def weight_variable(shape):
    initializer = tf.initializers.GlorotUniform()  # Xavier initializer
    initial = initializer(shape=shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def relu(c, b):
    return tf.nn.relu(c + b)

def convgradprop(x, DY, W, stride):
        linear_operation = lambda input_tensor, weights, s: conv2d(input_tensor, weights, s)
        Y = linear_operation(x, W, stride)
        return tf.gradients(Y, x, DY)[0]

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

def lrp(in_, x, R, act_, W, stride):
    R = tf.reshape(R, tf.shape(act_))
    Z = act_ + 1e-9
    S = R / Z
    if stride is None:
        C = tf.reduce_sum(tf.expand_dims(W, 0) * tf.expand_dims(S, 1), -1)
    else:
        C = convgradprop(x, S, W, stride)
    R = in_ * C
    return R

class ConvModel(object):
    ''' Implements the ConvNet model from the NVIDIA paper '''

    def __init__(self, dropout_prob=0.2, batch_norm=False, whitening=False, is_training=True):
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 66, 200, 3], name='x')
        self.y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.drop = 0.0
        # IN: img (3, 66, 200)      OUT: 1, 31, 98, 24
        self.W_conv1 = weight_variable([5, 5, 3, 24])
        self.b_conv1 = bias_variable([24])
        self.s_conv1 = 2
        self.hconv1 = conv2d(self.x, self.W_conv1, self.s_conv1) + self.b_conv1
        self.h_conv1 = tf.nn.relu(self.hconv1)

        # if batch_norm:
        #    self.h_conv1 = tf.contrib.layers.batch_norm(self.h_conv1, is_training=is_training, trainable=True)

        # IN: 1, 31, 98, 24     OUT: 1, 14, 47, 36
        self.W_conv2 = weight_variable([5, 5, 24, 36])
        self.b_conv2 = bias_variable([36])
        self.s_conv2 = 2
        self.hconv2 = conv2d(self.h_conv1, self.W_conv2, self.s_conv2) + self.b_conv2
        self.h_conv2 = tf.nn.relu(self.hconv2)

        # IN: 1, 14, 47, 36     OUT: 1, 5, 22, 48
        self.W_conv3 = weight_variable([5, 5, 36, 48])
        self.b_conv3 = bias_variable([48])
        self.s_conv3 = 2
        self.hconv3 = conv2d(self.h_conv2, self.W_conv3, self.s_conv3) + self.b_conv3
        self.h_conv3 = tf.nn.relu(self.hconv3)
        # if batch_norm:
        #    self.h_conv3 = tf.contrib.layers.batch_norm(self.h_conv3, is_training=is_training, trainable=True)

        # IN: 1, 5, 22, 48     OUT: 1, 3, 20, 64
        self.W_conv4 = weight_variable([3, 3, 48, 64])
        self.b_conv4 = bias_variable([64])
        self.s_conv4 = 1
        self.hconv4 = conv2d(self.h_conv3, self.W_conv4, self.s_conv4) + self.b_conv4
        self.h_conv4 = tf.nn.relu(self.hconv4)

        # IN: 1, 3, 20, 64         OUT: 1, 1, 18, 64
        self.W_conv5 = weight_variable([3, 3, 64, 64])
        self.b_conv5 = bias_variable([64])
        self.s_conv5 = 1
        self.hconv5 = conv2d(self.h_conv4, self.W_conv5, self.s_conv5) + self.b_conv5
        self.h_conv5 = tf.nn.relu(self.hconv5)
        # if batch_norm:
        #    self.h_conv5 = tf.contrib.layers.batch_norm(self.h_conv5, is_training=is_training, trainable=True)
        #self.W_flat = weight_variable([1152, 1])

        self.W_fc1 = weight_variable([1152, 1164])
        self.b_fc1 = bias_variable([1164])
        # IN: 1, 1, 18, 64      OUT: 1152
        self.h_conv5_flat = tf.reshape(self.h_conv5, [-1, 1152])

        # IN: 1152         OUT: 1164
        self.hfc1 = tf.matmul(self.h_conv5_flat, self.W_fc1) + self.b_fc1
        self.h_fc1 = tf.nn.relu(self.hfc1, name='fc1')
        # if batch_norm:
        #    self.h_fc1 = tf.contrib.layers.batch_norm(self.h_fc1, is_training=is_training, trainable=True)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.drop)

        # IN: 1164        OUT: 100
        self.W_fc2 = weight_variable([1164, 100])
        self.b_fc2 = bias_variable([100])
        self.hfc2 = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        self.h_fc2 = tf.nn.relu(self.hfc2, name='fc2')
        # if batch_norm:
        #    self.h_fc2 = tf.contrib.layers.batch_norm(self.h_fc2, is_training=is_training, trainable=True)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.drop)

        # IN: 100         OUT: 50
        self.W_fc3 = weight_variable([100, 50])
        self.b_fc3 = bias_variable([50])
        self.hfc3 = tf.matmul(self.h_fc2_drop, self.W_fc3) + self.b_fc3
        self.h_fc3 = tf.nn.relu(self.hfc3, name='fc3')
        # if batch_norm:
        #    self.h_fc3 = tf.contrib.layers.batch_norm(self.h_fc3, is_training=is_training, trainable=True)
        self.h_fc3_drop = tf.nn.dropout(self.h_fc3, self.drop)

        # IN: 50         OUT: 10
        self.W_fc4 = weight_variable([50, 10])
        self.b_fc4 = bias_variable([10])
        self.hfc4 = tf.matmul(self.h_fc3_drop, self.W_fc4) + self.b_fc4
        self.h_fc4 = tf.nn.relu(self.hfc4, name='fc4')
        # if batch_norm:
        #    self.h_fc4 = tf.contrib.layers.batch_norm(self.h_fc4, is_training=is_training, trainable=True)
        self.h_fc4_drop = tf.nn.dropout(self.h_fc4, self.drop)

        # IN: 10         OUT: 1
        self.W_fc5 = weight_variable([10, 1])
        self.b_fc5 = bias_variable([1])
        self.y = tf.multiply(tf.atan(tf.matmul(self.h_fc4_drop, self.W_fc5) + self.b_fc5), 2, name='y')


        self.fc2 = self.h_fc2
        self.fc3 = self.h_fc3

    def relprob(self, sess, R, x):
        sessG = sess
        in_ = sess.run(self.hfc4, feed_dict={self.x: x}) #10
        act_ = sess.run(self.y, feed_dict={self.x: x}) #1
        R = lrp(in_, x, R, act_, self.W_fc5, None) #10 .. W (10, 1)
        print("R", R.eval(session=sess))
        in_ = sess.run(self.hfc3, feed_dict={self.x: x}) #50
        act_ = sess.run(self.hfc4, feed_dict={self.x: x}) #10
        R = lrp(in_, x, R, act_, self.W_fc4, None) #50 .. W (50, 10)
        print("R", R.eval(session=sess))
        in_ = sess.run(self.hfc2, feed_dict={self.x: x}) #100
        act_ = sess.run(self.hfc3, feed_dict={self.x: x}) #50
        R = lrp(in_, x, R, act_, self.W_fc3, None) #100 .. W(100, 50)
        print("R", R.eval(session=sess))
        in_ = sess.run(self.hfc1, feed_dict={self.x: x}) #1164
        act_ = sess.run(self.hfc2, feed_dict={self.x: x}) #100
        R = lrp(in_, x, R, act_, self.W_fc2, None) #1164 .. W(1164, 100)
        print("R", R.eval(session=sess))
        in_ = sess.run(tf.reshape(self.hconv5, tf.shape(self.h_conv5_flat)), feed_dict={self.x: x}) #1152
        act_ = sess.run(self.hfc1, feed_dict={self.x: x}) #1164
        R = lrp(in_, x, R, act_, self.W_fc1, None) #1152 .. W(1152, 1164)
        print("R", R.eval(session=sess))
        print("*****")
        #in_ = sess.run(self.hconv5, feed_dict={self.x: x}) #1, 18, 64
        #act_ = sess.run(self.h_conv5_flat, feed_dict={self.x: x}) #1152
        #W = weight_variable([18, 64, 1152])
        #W = self.W_conv5
        #sess.run(tf.compat.v1.global_variables_initializer())
        #R = lrp(in_, x, R, act_, W, None) #1, 18, 64 ... W (1152)
        #print("in", in_.shape)
        #print("act", act_.shape)
        #print("W_in", self.W_conv5.shape)
        #print("W_out", W.shape)
        #print("R.shape", R.eval(session=sess).shape)
        #print("R", R.eval(session=sess))
        #print("****")

        in_ = sess.run(self.hconv4, feed_dict={self.x: x}) #3, 20, 64
        act_ = sess.run(self.hconv5, feed_dict={self.x: x}) #1, 18, 64
        print("in", in_.shape)
        print("act", act_.shape)
        #print("Rin", R.eval(session=sess).shape)
        print("Eval")
        W = weight_variable([1, 1, 18, 64])
        W = self.W_conv5
        #R = lrp(in_, R, act_, W, self.s_conv5) #3, 20, 64 .. W ()
        sess.run(self.x, feed_dict={self.x: x})
        R = lrp(self.hconv4, self.hconv4, R, act_, W, self.s_conv5) #3, 20, 64 .. W (3, 3, 64, 64)
        print("R", sess.run(R, feed_dict={self.x: x}))
        #R = lrp(in_, R, act_, self.W_conv5)
        #print("R", R.eval(session=sess).shape)
        in_ = sess.run(self.hconv3, feed_dict={self.x: x})
        act_ = sess.run(self.hconv4, feed_dict={self.x: x})
        R = lrp(self.hconv3, self.hconv3, R, act_, self.W_conv4, self.s_conv4)
        print("R", sess.run(R, feed_dict={self.x: x}))
        in_ = sess.run(self.hconv2, feed_dict={self.x: x})
        act_ = sess.run(self.hconv3, feed_dict={self.x: x})
        R = lrp(self.hconv2, self.hconv2, R, act_, self.W_conv3, self.s_conv3)
        print("R", sess.run(R, feed_dict={self.x: x}))
        in_ = sess.run(self.hconv1, feed_dict={self.x: x})
        act_ = sess.run(self.hconv2, feed_dict={self.x: x})
        R = lrp(self.hconv1, self.hconv1, R, act_, self.W_conv2, self.s_conv2)
        print("R", sess.run(R, feed_dict={self.x: x}))
        in_ = sess.run(self.x, feed_dict={self.x: x})
        act_ = sess.run(self.hconv1, feed_dict={self.x: x})
        R = lrp(self.x, self.x, R, act_, self.W_conv1, self.s_conv1)
        print("R", sess.run(R, feed_dict={self.x: x}))

    def update(self, x, y_):
        self.y_ = y_
        self.x = x


class ConvModelX(object):
    ''' Implements the ConvNet model from the NVIDIA paper '''

    def __init__(self, dropout_prob=0.2, batch_norm=False, whitening=False, is_training=True):
        x = tf.compat.v1.placeholder(tf.float32, shape=[None, 66, 200, 3], name='x')
        keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
        self.x_image = x
        self.keep_prob = keep_prob

        y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.conv1 = conv2d(x, weight_variable([5, 5, 3, 24]), 2)
        self.h_conv1 = relu(self.conv1, bias_variable([24]))
        # if batch_norm:
        #    self.h_conv1 = tf.contrib.layers.batch_norm(self.h_conv1, is_training=is_training, trainable=True)
        self.conv2 = conv2d(self.h_conv1, weight_variable([5, 5, 24, 36]), 2)
        self.h_conv2 = relu(self.conv2, bias_variable([36]))
        self.conv3 = conv2d(self.h_conv2, weight_variable([5, 5, 36, 48]), 2)
        self.h_conv3 = relu(self.conv3, bias_variable([48]))
        # if batch_norm:
        #    self.h_conv3 = tf.contrib.layers.batch_norm(self.h_conv3, is_training=is_training, trainable=True)

        self.conv4 = conv2d(self.h_conv3, weight_variable([3, 3, 48, 64]), 1)
        self.h_conv4 = relu(self.conv4, bias_variable([64]))
        self.conv5 = conv2d(self.h_conv4, weight_variable([3, 3, 64, 64]), 1)
        self.h_conv5 = relu(self.conv5, bias_variable([64]))
        # if batch_norm:
        #    self.h_conv5 = tf.contrib.layers.batch_norm(self.h_conv5, is_training=is_training, trainable=True)
        self.conv5_reshape = tf.reshape(self.h_conv5, [-1, 1152])
        self.h_fc1 = relu(tf.matmul(self.conv5_reshape, weight_variable([1152, 1164])),
                          bias_variable([1164]))
        # if batch_norm:
        #    self.h_fc1 = tf.contrib.layers.batch_norm(self.h_fc1, is_training=is_training, trainable=True)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.h_fc2 = relu(tf.matmul(self.h_fc1_drop, weight_variable([1164, 100])), bias_variable([100]))
        # if batch_norm:
        #    self.h_fc2 = tf.contrib.layers.batch_norm(self.h_fc2, is_training=is_training, trainable=True)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)

        self.h_fc3 = relu(tf.matmul(self.h_fc2_drop, weight_variable([100, 50])), bias_variable([50]))
        # if batch_norm:
        #    self.h_fc3 = tf.contrib.layers.batch_norm(self.h_fc3, is_training=is_training, trainable=True)
        self.h_fc3_drop = tf.nn.dropout(self.h_fc3, self.keep_prob)

        self.h_fc4 = relu(tf.matmul(self.h_fc3_drop, weight_variable([50, 10])), bias_variable([10]))
        # if batch_norm:
        #    self.h_fc4 = tf.contrib.layers.batch_norm(self.h_fc4, is_training=is_training, trainable=True)
        self.h_fc4_drop = tf.nn.dropout(self.h_fc4, self.keep_prob)

        y = tf.multiply(tf.atan(tf.matmul(self.h_fc4_drop, weight_variable([10, 1])) + bias_variable([1])), 2, name='y')

        self.x = x
        self.y_ = y_
        self.y = y
        self.fc2 = self.h_fc2
        self.fc3 = self.h_fc3


if __name__ == '__main__':
    print(ConvModel())
