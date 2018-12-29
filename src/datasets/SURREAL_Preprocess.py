from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import makedirs, listdir, path
from os.path import join, exists
from src.datasets.smpl_np import SMPLModel
from src.datasets.common import *

import cv2
import imageio
import numpy as np
import tensorflow as tf
imageio.plugins.ffmpeg.download()


tf.app.flags.DEFINE_string('tfRecord_dir',
                           '/home/cha/SamSung/DataSet/tf_datasets/SURREAL_Basic',
                           'tf set output')
tf.app.flags.DEFINE_string('base_dir',
                           '/home/cha/SamSung/DataSet/data/SURREAL',
                           'base dir')
tf.app.flags.DEFINE_string('smpl_dir',
                           '/home/cha/SamSung/hmr/src/datasets/smpl_man.pkl',
                           'smpl_directory')
tf.app.flags.DEFINE_string('tf_smpl_dir',
                           '/home/cha/SamSung/hmr/src/datasets/smpl_man.pkl',
                           'smpl_directory')
tf.app.flags.DEFINE_integer('num_shards', 500,
                            'Number of shards in training TFRecord files')
FLAGS = tf.app.flags.FLAGS


def process_surreal(base_dir, out_dir, smpl_dir, split):
    smpl_np = SMPLModel(smpl_dir)
    base_dir = join(base_dir, split)
    out_dir = join(out_dir, 'train')
    coder =ImageCoder()

    t = split.split('/')
    # run0, run1, run2
    tmp = t[0] + '_' + t[1]
    tf_filename = join(out_dir, tmp +'.tfrecord')

    for f2 in sorted(listdir(base_dir)):
        print(f2)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            for f3 in sorted(listdir(join(base_dir, f2))):
                if not f3.endswith('.mp4') and not f3.endswith('.mat'):
                    vid_parse = path.splitext(path.basename(f3))[0]
                    folder_path = join(base_dir, f2, vid_parse)

                    matname = path.splitext(path.basename(f3))[0] + '_info.mat'
                    matname = join(base_dir, f2, matname)

                    camLoc, gt3ds, gt2ds, poses, shapes, zrot_vid, camDist = read_mat(matname)
                    cam = np.ones([3])
                    cam[0] = camDist
                    cam[1] = camLoc[0]
                    cam[2] = camLoc[1]

                    for i, f4 in enumerate(sorted(listdir(join(base_dir, f2, f3, folder_path)))):
                        img_path = join(base_dir, f2, f3, folder_path, 'frame_{0}.jpg'.format(i+1))
                        gt2d = np.squeeze(gt2ds[:, :, i])
                        gt3d = np.squeeze(gt3ds[:, :, i])
                        pose = np.squeeze(poses[:, i])
                        shape = np.squeeze(shapes[:, i])

                        global count

                        count += add_tfrecord(img_path, tf_filename, gt2d, gt3d, cam, pose, shape, coder, writer)


def add_tfrecord(im_path, filename, gt2d, gt3d, cam, pose, shape, coder, writer):

    # check the image path
    if not exists(im_path):
        print('!!--%s doesnt exist! Skipping..--!!' % im_path)
        return False

    image = cv2.imread(im_path)
    # BGR to RGB color format

    assert image.shape[2] == 3

    # check the state of image
    flags, length, mid = check_good_data(image, gt2d, gt3d)
    if not flags:
        #print('wrong data')
        return 0

    #draw_skeleton_lsp(image, gt2d)

    # change joints2D 24 -> 14
    # img cropped to rectangle shape(properly)
    image, gt2d, cam = img_crop(image, gt2d, cam, length, mid)

    gt2d = j2d_24_to_14(gt2d)
    gt3d = j3d_24_to_14(gt3d)

    gt3d = np.transpose(gt3d)
    gt2d = np.transpose(gt2d)

    min_pt = np.min(gt2d, axis=0)
    max_pt = np.max(gt2d, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    image_scaled, scale_factors = resize_img(image, scale)
    height, width = image_scaled.shape[:2]
    joints_scaled = np.copy(gt2d)
    joints_scaled[:, 0] *= scale_factors[0]
    joints_scaled[:, 1] *= scale_factors[1]
    center_scaled = np.round(center * scale_factors).astype(np.int)
    # scale camera: Flength, px, py
    cam_scaled = np.copy(cam)
    cam_scaled[0] *= scale
    cam_scaled[1] *= scale_factors[0]
    cam_scaled[2] *= scale_factors[1]

    # Crop 300x300 around the center
    margin = 150
    start_pt = np.maximum(center_scaled - margin, 0).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], width)
    end_pt[1] = min(end_pt[1], height)
    image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[
        0], :]
    # Update others too.
    joints_scaled[:, 0] -= start_pt[0]
    joints_scaled[:, 1] -= start_pt[1]
    center_scaled -= start_pt
    # Update principal point:
    cam_scaled[1] -= start_pt[0]
    cam_scaled[2] -= start_pt[1]
    height, width = image_scaled.shape[:2]

    # Fix units: mm -> meter
    gt3d = gt3d / 1000.
    # gt3d = gt3d

    # Encode image:
    image_data_scaled = coder.encode_jpeg(image_scaled)
    label = np.vstack([joints_scaled.T,
                       np.ones((1, joints_scaled.shape[0]))])
    example = convert_to_example_wmosh(image_data_scaled, filename,
                                       height, width, label, center,
                                       gt3d, pose, shape, scale_factors,
                                       start_pt, cam)

    writer.write(example.SerializeToString())

    return 1

def check_good_data(image, joints2D, joints3D):
    # margin for image and crop the image to rectangle
    # joints2D(12) is neck
    # joints2D(2) is right foot
    margin = abs(joints2D[1, 12] - joints2D[1, 2])
    margin /= 3.

    x_len, y_len, mid = calcul_mid(joints2D, margin)
    length = max(x_len, y_len)

    # if value over the boundary can't decode
    if mid[1] - length < 0 or mid[0] - length < 0 \
            or length + mid[0] > 320 or length + mid[1] > 240:
        return False, length, mid
    else:
        return True, length, mid


def main(unused_argv):
    if not exists(join(FLAGS.tfRecord_dir, 'test')):
        makedirs(join(FLAGS.tfRecord_dir, 'test'))
    if not exists(join(FLAGS.tfRecord_dir, 'train')):
        makedirs(join(FLAGS.tfRecord_dir, 'train'))
    if not exists(join(FLAGS.tfRecord_dir, 'val')):
        makedirs(join(FLAGS.tfRecord_dir, 'val'))

    global count
    count = 0

    # test
    #process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'test/run0')
    #process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'test/run1')
    #process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'test/run2')

    process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'train/run0')
    process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'train/run1')
    process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'train/run2')

    # validation
    #process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'val/run0')
    #process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'val/run1')
    #process_surreal(FLAGS.base_dir, FLAGS.tfRecord_dir, FLAGS.smpl_dir, 'val/run2')

    print('How many dataset do i make? : ', count)


if __name__ == '__main__':
   tf.app.run()
