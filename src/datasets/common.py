from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities.
    Taken from
    https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py
    """

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(
            image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)
        self._encode_jpeg = tf.image.encode_jpeg(
            self._encode_jpeg_data, format='rgb')

        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(
            self._decode_png_data, channels=3)

        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(self._encode_png_data)

    def png_to_jpeg(self, image_data):
        return self._sess.run(
            self._png_to_jpeg, feed_dict={
                self._png_data: image_data
            })

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_jpeg(self, image):
        image_data = self._sess.run(
            self._encode_jpeg, feed_dict={
                self._encode_jpeg_data: image
            })
        return image_data

    def encode_png(self, image):
        image_data = self._sess.run(
            self._encode_png, feed_dict={
                self._encode_png_data: image
            })
        return image_data

    def decode_png(self, image_data):
        image = self._sess.run(
            self._decode_png, feed_dict={
                self._decode_png_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list) and not isinstance(value, np.ndarray):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_example_wmosh(image_data, image_path, height, width, label,
                             center, gt3d, pose, shape, scale_factors,
                             start_pt, cam):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      image_path: string, path to this image file
      labels: 3 x 14 joint location + visibility
      height, width: integers, image shapes in pixels.
      center: 2 x 1 center of the tight bbox
      gt3d: 14x3 3D joint locations
      scale_factors: 2 x 1, scale factor used to scale image.
      start_pt: the left corner used to crop the _scaled_ image to 300x300
      cam: (3,), [f, px, py] intrinsic camera parameters.
    Returns:
      Example proto
    """
    from os.path import basename
    image_format = 'JPEG'
    if label.shape[0] != 3:
        label = label.T
    if label.shape[1] > 14:
        print('This shouldnt be happening')
        import ipdb
        ipdb.set_trace()
    if pose is None:
        has_3d = 0
        # Use -1 to save.
        pose = -np.ones(72)
        shape = -np.ones(10)
    else:
        has_3d = 1

    example = tf.train.Example(
        features=tf.train.Features(feature={
            'image/height':
                int64_feature(height),
            'image/width':
                int64_feature(width),
            'image/center':
                int64_feature(center.astype(np.int)),
            'image/x':
                float_feature(label[0, :].astype(np.float)),
            'image/y':
                float_feature(label[1, :].astype(np.float)),
            'image/visibility':
                int64_feature(label[2, :].astype(np.int)),
            'image/format':
                bytes_feature(tf.compat.as_bytes(image_format)),
            'image/filename':
                bytes_feature(tf.compat.as_bytes(basename(image_path))),
            'image/encoded':
                bytes_feature(tf.compat.as_bytes(image_data)),
            'mosh/pose':
                float_feature(pose.astype(np.float)),
            'mosh/shape':
                float_feature(shape.astype(np.float)),
            'mosh/gt3d':
                float_feature(gt3d.ravel().astype(np.float)),
            'meta/scale_factors':
                float_feature(np.array(scale_factors).astype(np.float)),
            'meta/crop_pt':
                int64_feature(start_pt.astype(np.int)),
            'meta/has_3d':
                int64_feature(has_3d),
            'image/cam':
                float_feature(cam.astype(np.float)),
        }))

    return example


def resize_img(img, scale_factor):
    import cv2
    import numpy as np
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def img_crop(image, joints2D, cam, length, mid):
    # mid[0] is x coord
    # mid[1] is y coord
    # image is (y, x)
    # start_pt is (x, y)
    start_pt = [mid[0]-length, mid[1]-length]
    end_pt = [mid[0]+length, mid[1]+length]
    crop_img = image[start_pt[1]: end_pt[1], start_pt[0]: end_pt[0]]

    # translate 2d joints
    for i in range(24):
        joints2D[0, i] -= start_pt[0]
        joints2D[1, i] -= start_pt[1]
        cam[1] -= start_pt[0]
        cam[2] -= start_pt[1]

    return crop_img, joints2D, cam


def calcul_mid(joints2D, margin):
    # min_coor is (x, y)
    # image is (y, x)!!
    # min, max coordinate find to bounding box
    # length is value for crop size!
    min_coor = [min(joints2D[0, :]) - margin,
                min(joints2D[1, :]) - margin - margin / 5.]
    min_coor = list(map(int, min_coor))
    max_coor = [max(joints2D[0, :]) + margin,
                max(joints2D[1, :]) + margin - margin / 3.]
    max_coor = list(map(int, max_coor))

    y_len = abs(min_coor[0] - max_coor[0])
    x_len = abs(min_coor[1] - max_coor[1])
    x_len = int(x_len / 2.)
    y_len = int(y_len / 2.)
    mid = [(min_coor[0] + max_coor[0]) / 2., (min_coor[1] + max_coor[1]) / 2.]
    mid = list(map(int, mid))

    return x_len, y_len, mid


def read_images_from_tfrecords(tf_path, img_size=224, sess=None):
    """
    Returns image, kp, and gt3d from the tf_paths
    This returns a preprocessed image, cropped around img_size.
    """
    from time import time
    from os.path import exists
    if not exists(tf_path):
        print('%s doesnt exist!' % tf_path)
        exit(1)

    if sess is None:
        sess = tf.Session()

    t0 = time()
    all_images, all_kps, all_gt3ds = [], [], []

    itr = 0

    # Decode op graph
    image_data_pl = tf.placeholder(dtype=tf.string)
    decode_op = tf.image.decode_jpeg(image_data_pl)

    for serialized_ex in tf.python_io.tf_record_iterator(tf_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_ex)
        image_data = example.features.feature['image/encoded'].bytes_list.value[0]
        image = sess.run(decode_op, feed_dict={image_data_pl: image_data})

        x = example.features.feature['image/x'].float_list.value
        y = example.features.feature['image/y'].float_list.value
        vis = example.features.feature['image/visibility'].int64_list.value
        center = example.features.feature['image/center'].int64_list.value

        x = np.array(x)
        y = np.array(y)
        vis = np.array(vis, dtype='bool')
        center = np.array(center)

        # Crop img_size.
        # Pad in case.
        margin = int(img_size / 2)
        image_pad = np.pad(image, ((margin,), (margin,), (0,)), mode='edge')

        # figure out starting point
        start_pt = center
        end_pt = center + 2 * margin

        x_crop = x + margin - start_pt[0]
        y_crop = y + margin - start_pt[1]
        kp_crop = np.vstack([x_crop, y_crop])
        kp_final = 2 * (kp_crop / img_size) - 1
        kp_final = np.vstack((vis * kp_final, vis)).T
        # crop:
        crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        # Note: This says mosh but gt3d is the gt H3.6M joints & not from mosh.
        gt3d = example.features.feature['mosh/gt3d'].float_list.value
        gt3d = np.array(gt3d).reshape(-1, 3)

        all_images.append(crop)
        all_kps.append(kp_final)
        all_gt3ds.append(gt3d)

        itr += 1

    images = np.stack(all_images)
    kps = np.stack(all_kps)
    gt3ds = np.stack(all_gt3ds)

    print('Read %d images, %g secs' % (images.shape[0], time() - t0))

    return images, kps, gt3ds

def j2d_24_to_14(joints):
    '''
    smpl -> lsp
    0   ->
    1   ->  3
    2   ->  2
    3   ->
    4   ->  4
    5   ->  1
    6   ->
    7   ->  5
    8   ->  0
    9   ->
    10  ->
    11  ->
    12  ->  12
    13  ->
    14  ->
    15  ->  13
    16  ->  9
    17  ->  8
    18  ->  10
    19  ->  7
    20  ->  11
    21  ->  6
    22  ->
    23  ->
    '''
    lsp_j = np.ones((2, 14))
    for i in range(2):
        lsp_j[i, 0] = joints[i, 8]
        lsp_j[i, 1] = joints[i, 5]
        lsp_j[i, 2] = joints[i, 2]
        lsp_j[i, 3] = joints[i, 1]
        lsp_j[i, 4] = joints[i, 4]
        lsp_j[i, 5] = joints[i, 7]
        lsp_j[i, 6] = joints[i, 21]
        lsp_j[i, 7] = joints[i, 19]
        lsp_j[i, 8] = joints[i, 17]
        lsp_j[i, 9] = joints[i, 16]
        lsp_j[i, 10] = joints[i, 18]
        lsp_j[i, 11] = joints[i, 20]
        lsp_j[i, 12] = joints[i, 12]
        lsp_j[i, 13] = joints[i, 15]

    return lsp_j


def j3d_24_to_14(joints3D):
    joints3D_14 = np.ones((3, 14))
    for i in range(3):
        joints3D_14[i, 0] = joints3D[i, 8]
        joints3D_14[i, 1] = joints3D[i, 5]
        joints3D_14[i, 2] = joints3D[i, 2]
        joints3D_14[i, 3] = joints3D[i, 1]
        joints3D_14[i, 4] = joints3D[i, 4]
        joints3D_14[i, 5] = joints3D[i, 7]
        joints3D_14[i, 6] = joints3D[i, 21]
        joints3D_14[i, 7] = joints3D[i, 19]
        joints3D_14[i, 8] = joints3D[i, 17]
        joints3D_14[i, 9] = joints3D[i, 16]
        joints3D_14[i, 10] = joints3D[i, 18]
        joints3D_14[i, 11] = joints3D[i, 20]
        joints3D_14[i, 12] = joints3D[i, 12]
        joints3D_14[i, 13] = joints3D[i, 15]

    return joints3D_14


def draw_skeleton_lsp(image, joints, radius=2):
    if joints.shape[0] > 2:
        joints = np.transpose(joints)

    colors = {'purple': np.array([118, 42, 131]),
              'green': np.array([0, 255, 0])}

    image = image.copy()
    input_is_float = False
    print(joints.shape)

    joints = joints.astype(int)

    for num in range(14):
        cv2.circle(image, (joints[0, num], joints[1, num]),
                   radius, colors['purple'], -1)

    # cv2.rectangle(image, tuple(min_coor), tuple(max_coor), colors['green'], 2)

    print(image.shape)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    return image


def get_data(vid, frame, joints3D_vid, joints2D_vid, shape_vid, pose_vid, zrot_vid):
    # read a image frame
    image = vid.get_data(frame)
    # BGR to RGB color format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # label processing
    joints3D = joints3D_vid[:, :, frame]
    np.squeeze(joints3D)
    pose = pose_vid[:, frame]
    np.squeeze(pose)
    shape = shape_vid[:, frame]
    np.squeeze(shape)
    joints2D = joints2D_vid[:, :, frame]
    zrot = zrot_vid[frame, :]
    np.squeeze(zrot)

    return image, joints3D, joints2D, shape, pose, zrot


def load_mat(fname, idx):
    import scipy.io as sio
    res = sio.loadmat(fname)
    return res[idx]


def read_mat(matname):
    camLoc = load_mat(matname, 'camLoc')
    joints3D_vid = load_mat(matname, 'joints3D')
    joints2D_vid = load_mat(matname, 'joints2D')
    pose_vid = load_mat(matname, 'pose')
    shape_vid = load_mat(matname, 'shape')
    zrot_vid = load_mat(matname, 'zrot')
    camDist = load_mat(matname, 'camDist')

    return camLoc, joints3D_vid, joints2D_vid, pose_vid, shape_vid, zrot_vid, camDist


def get_intrinsic():
    res_x_px = 320
    res_y_px = 240
    f_mm = 60
    sensor_w_mm = 32
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px

    scale = 1
    skew = 0
    pixel_aspect_ratio = 1
    fx_px = f_mm * res_x_px * scale / sensor_w_mm
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

    u = res_x_px * scale / 2
    v = res_y_px * scale / 2

    K = np.array([fx_px, skew, u, 0, fy_px, v, 0, 0, 1])
    K = np.reshape(K, (3, 3))

    return K


def get_extrinsic(T):
    R_world2bcam = np.array([[0, 0, 1],
                             [0, -1, 0],
                             [-1, 0, 0]])

    T_world2bcam = -1 * R_world2bcam
    T_world2bcam = np.matmul(T_world2bcam, T)
    T_world2bcam = np.reshape(T_world2bcam, (3, 1))

    R_bcam2cv = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])

    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    RT = np.concatenate((R_world2cv, T_world2cv), axis=1)

    RT[0, :] = RT[0, :] * -1
    RT[2, :] = RT[2, :] * -1

    '''
    [   0   0   -1  -1.9293
        0   1   0   0.9839
        -1  0   0   6.7456  ]
    '''
    return RT


