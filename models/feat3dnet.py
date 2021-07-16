import logging
import tensorflow as tf
from models.layers import conv2d

def feature_detection_module(net, is_training, use_bn=True):
    mlp = [64, 128, 256]
    mlp2 = [128, 64]
    end_points = {}

    # Pre pooling MLP
    for i, num_out_channel in enumerate(mlp):
        net = conv2d(net, num_out_channel, kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                            bn=use_bn, is_training=is_training,
                            scope='conv%d' % (i), reuse=False, )

    # Max Pool
    net = tf.reduce_max(net, axis=[2], keep_dims=True)

    # Max pooling MLP
    if mlp2 is None: mlp2 = []
    for i, num_out_channel in enumerate(mlp2):
        net = conv2d(net, num_out_channel, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=use_bn, is_training=is_training,
                            scope='conv_post_%d' % (i))

    # Attention and orientation regression
    attention = conv2d(net, 1, [1, 1], stride=[1, 1], padding='VALID',
                       activation=tf.nn.softplus, bn=False, scope='attention', reuse=False)
    attention = tf.squeeze(attention, axis=[2, 3])

    orientation_xy = conv2d(net, 2, [1, 1], stride=[1, 1], padding='VALID',
                            activation=None, bn=False, scope='orientation', reuse=False)
    orientation_xy = tf.squeeze(orientation_xy, axis=2)
    orientation_xy = tf.nn.l2_normalize(orientation_xy, dim=2, epsilon=1e-8)
    orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

    return attention, orientation, end_points

def feature_extraction_module():
    # Descriptor extraction: Extract descriptors for each cluster
    mlp = [32, 64]
    mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
    mlp3 = [self.param['feature_dim']]

    # Extracts descriptors
    l1_xyz, l1_points, l1_idx, end_points = pointnet_sa_module(l0_xyz, l0_points, 512, radius, num_samples,
                                                   mlp=mlp, mlp2=mlp2, mlp3=mlp3,
                                                   is_training=is_training, scope='layer1',
                                                   bn=use_bn, bn_decay=None,
                                                   keypoints=keypoints, orientations=orientations, normalize_radius=True,
                                                   final_relu=False)

    xyz = l1_xyz
    features = tf.nn.l2_normalize(l1_points, dim=2, epsilon=1e-8)

    return xyz, features, end_points

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, mlp3, is_training, scope, bn=True, bn_decay=None,
                       tnet_spec=None, knn=False, use_xyz=True,
                       keypoints=None, orientations=None, normalize_radius=True, final_relu=True):
    """ PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP

    Args:
        xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
        points (tf.Tensor): (batch_size, ndataset, num_channel)
        npoint (int32): #points sampled in farthest point sampling
        radius (float): search radius in local region
        nsample (int): Maximum points in each local region
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        is_training (tf.placeholder): Indicate training/validation
        scope (str): name scope
        bn (bool): Whether to perform batch normalizaton
        bn_decay: Decay schedule for batch normalization
        tnet_spec: Unused in Feat3D-Net. Set to None
        knn: Unused in Feat3D-Net. Set to False
        use_xyz: Unused in Feat3D-Net. Set to True
        keypoints: If provided, cluster centers will be fixed to these points (npoint will be ignored)
        orientations (tf.Tensor): Containing orientations from the detector
        normalize_radius (bool): Whether to normalize coordinates [True] based on cluster radius.
        final_relu: Whether to use relu as the final activation function

    Returns:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
        idx: (batch_size, npoint, nsample) int32 -- indices for local regions

    """

    with tf.variable_scope(scope) as sc:
        if npoint is None:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz, end_points = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec,
                                                                                 knn, use_xyz,
                                                                                 keypoints=keypoints,
                                                                                 orientations=orientations,
                                                                                 normalize_radius=normalize_radius)

        for i, num_out_channel in enumerate(mlp):
            new_points = conv2d(new_points, num_out_channel, kernel_size=[1, 1], stride=[1, 1], padding='VALID',
                                bn=bn, is_training=is_training,
                                scope='conv%d' % (i), reuse=False, )

        # Max pool
        pooled = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        pooled_expand = tf.tile(pooled, [1, 1, new_points.shape[2], 1])

        # Concatenate
        new_points = tf.concat((new_points, pooled_expand), axis=3)

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = conv2d(new_points, num_out_channel, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=bn, is_training=is_training,
                                scope='conv_mid_%d' % (i), bn_decay=bn_decay,
                                activation=tf.nn.relu if (final_relu or i < len(mlp2) - 1) else None)

        # Max pool again
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

        if mlp3 is None:
            mlp3 = []
        for i, num_out_channel in enumerate(mlp3):
            new_points = conv2d(new_points, num_out_channel, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=bn, is_training=is_training,
                                scope='conv_post_%d' % (i), bn_decay=bn_decay,
                                activation=tf.nn.relu if (final_relu or i < len(mlp3) - 1) else None)
        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points, idx, end_points


class Feat3dNet:
    def __init__(self, param=None):
        """ Constructor: Sets the parameters for 3DFeat-Net

        Args:
            param:    Python dict containing the algorithm parameters. It should contain the
                      following fields (square brackets denote paper's parameters':
                      'NoRegress': Whether to skip regression of the keypoint orientation.
                                   [False] (i.e. regress)
                      'BaseScale': Cluster radius. [2.0] (as in the paper)
                      'Attention': Whether to predict the attention. [True]
                      'num_clusters': Number of clusters [512]
                      'num_samples': Maximum number of points per cluster [64]
                      'margin': Triplet loss margin [0.2]
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

    def get_model(self, point_cloud, is_training, use_bn=True):
        """ Constructs the core 3DFeat-Net model.

        Args:
            point_cloud (tf.Tensor): Input point clouds of size (batch_size, ndataset, 3).
            is_training (tf.placeholder): Set to true only if training, false otherwise
            use_bn (bool): Whether to perform batch normalization

        Returns:
            xyz, features, attention, end_points

        """
        end_points = {}

        with tf.variable_scope("detection") as sc:
            attention, orientation, end_points_temp = \
                feature_detection_module(point_cloud,
                                        is_training,
                                        use_bn=use_bn)

        end_points.update(end_points_temp)
        end_points['attention'] = attention
        end_points['orientation'] = orientation