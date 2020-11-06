from tensorflow.keras.layers import BatchNormalization, \
    LeakyReLU, Conv2D, ZeroPadding2D, Add
from tensorflow.keras.regularizers import l2


def conv(x, *args, **kwargs):
    new_kwargs = {'kernel_regularizer': l2(5e-4), 'use_bias': False,
                  'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    new_kwargs.update(kwargs)
    x = Conv2D(*args, **kwargs)(x)
    return x


def CBL(x, *args, **kwargs):
    x = conv(x, *args, **kwargs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def PCBL(x, num_filters):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = CBL(x, num_filters, (3, 3), strides=(2, 2))
    return x


def CBLR(x, num_filters):
    y = CBL(x, num_filters, (1, 1))
    y = CBL(y, num_filters*2, (3, 3))
    x = Add()([x, y])

    return x


def body(input):
    # Darknet53
    x = CBL(input, 32, (3, 3))
    n = [1, 2, 8, 8, 4]
    for i in range(5):
        x = PCBL(x, 2**(6+i))
        for _ in range(n[i]):
             x = CBLR(x)



    return output
