# Change your conda enviroment to Mobile-Vit (cntr + alt + s)
import tensorflow as tf
from keras.applications import imagenet_utils
from tensorflow.keras import layers
from tensorflow import keras
import cv2
import pdb
import matplotlib.pyplot as plt

# Values are from table 4.
patch_size = 4

# img size from paper --> 256
original_image_size = 256


# image_height = 150
# image_width = 1920

gray_img_channel = 1
rgb_channels = 3

expansion_factor = 4  # expansion factor for the MobileNetV2 blocks.


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(filters, kernel_size, strides, activation=tf.nn.swish, padding="same")
    return conv_layer(x)


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    """
    :return: MobileNetV2 block:
    point_wise_conv->BN->Swish->DepthWise->BN->Swish->pointwise->BN->if in.shape==out.shape then concat
    """
    m = layers.Conv2D(expanded_channels, kernel_size=1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, kernel_size=3))(m)
    m = layers.DepthwiseConv2D(kernel_size=3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, kernel_size=1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # layerNorm1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # create multihead attenstion-layer
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # skip connection 1
        x2 = layers.Add()([attention_output, x])
        # layerNorm2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x)
        # MLP
        x3 = mlp(x3, hidden_units=[x.shape[-1] * 2, x.shape[-1]], dropout_rate=0.1)
        # skip connection 2
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(local_features, filters=projection_dim, kernel_size=1, strides=strides)

    # Unfold into patches and then pass through transformer
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(local_features)

    # Global features
    global_features = transformer_block(non_overlapping_patches, num_blocks, projection_dim)

    # Fold into conv-like feature maps
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(global_features)

    # apply pointwise conv -> concat with the input features
    folded_feature_map = conv_block(folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides)
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse local and global features using a convolution layer
    local_global_features = conv_block(local_global_features, filters=projection_dim, strides=strides)

    return local_global_features


def create_mobilevit(mobile_vit_channels, mobile_vit_projection_dims, num_classes=1000, expansion_factor=4):
    # pdb.set_trace()
    inputs = keras.Input((original_image_size, original_image_size, rgb_channels))

    # initial conv-stem -> MV2 block
    x = conv_block(inputs, filters=mobile_vit_channels[0])
    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[0] * expansion_factor, output_channels=mobile_vit_channels[1])

    # Downsampling with MV2 block(stride=2) -> MV2 block(x2) -> Downsampling with MV2 block
    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[1] * expansion_factor, output_channels=mobile_vit_channels[2],
                                strides=2)

    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[2] * expansion_factor, output_channels=mobile_vit_channels[3])
    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[2] * expansion_factor, output_channels=mobile_vit_channels[3])

    # Fist Downsampling MV2 Block -> MobileViT block
    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[3] * expansion_factor, output_channels=mobile_vit_channels[4],
                                strides=2)
    x = mobilevit_block(x, num_blocks=2, projection_dim=mobile_vit_projection_dims[0])

    # Second Downsampling MV2 Block -> MobileViT block
    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[6] * expansion_factor, output_channels=mobile_vit_channels[7],
                                strides=2)
    x = mobilevit_block(x, num_blocks=4, projection_dim=mobile_vit_projection_dims[1])

    # Third Downsampling MV2 Block -> MobileViT block -> pointwise convolution
    x = inverted_residual_block(x, expanded_channels=mobile_vit_channels[8] * expansion_factor, output_channels=mobile_vit_channels[9],
                                strides=2)
    x = mobilevit_block(x, num_blocks=3, projection_dim=mobile_vit_projection_dims[2])
    x = conv_block(x, filters=mobile_vit_channels[10], kernel_size=1, strides=1)

    # classification head
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return create_mobilevit(channels, dims, num_classes=1000, expansion_factor=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return create_mobilevit(channels, dims, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return create_mobilevit(channels, dims, num_classes=1000)


mobilevit_xxs = mobilevit_xxs()
mobilevit_xxs.summary()
# test on random image
# img = tf.random.uniform((1, 150, 1920, 1))
# out = mobilevit_xxs(img)

# test on random image
# img = tf.random.uniform((1, 256, 256, 3))
# out = mobilevit_xxs(img)
