from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs

import numpy as np
import imageio
import cv2

import tensorflow as tf

tf.app.flags.DEFINE_string('base_dir',
                           '/home/cha/SamSung/DataSet/data/mpi_inf_3dhp',
                           'base dir')

tf.app.flags.DEFINE_string('output_dir',
                           '/home/cha/SamSung/DataSet/data/mpi_inf_3dhp',
                           'base dir')

FLAGS = tf.app.flags.FLAGS

def get_paths(base_dir, sub_id, seq_id):
    data_dir = join(base_dir, 'S%d' % sub_id, 'Seq%d' % seq_id)
    return data_dir

def video_to_img(base_dir, output_dir):

    sub_ids = range(1, 9)
    seq_ids = range(1, 3)
    cam_ids = [0, 1, 2, 4, 5, 6, 7, 8]

    for sub_id in sub_ids:
        print('%d is start' % sub_id)
        for seq_id in seq_ids:
            video_dir_o = join(get_paths(base_dir, sub_id, seq_id),
                             'imageSequence')
            output_dir_o = join(get_paths(base_dir, sub_id, seq_id),
                              'imageFrames')

            if not exists(output_dir):
                makedirs(output_dir)

            for cam_id in cam_ids:
                video_dir = join(video_dir_o, 'video_%d.avi' % cam_id)
                output_dir = join(output_dir_o, 'video_%d' % cam_id)
                print(video_dir)

                if not exists(output_dir):
                    makedirs(output_dir)

                # vid read
                vid = imageio.get_reader(video_dir, 'ffmpeg')

                for frame, im in enumerate(vid):
                    image = vid.get_data(frame)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    filename = join(output_dir, 'frame_%d.jpg' % (frame + 1))
                    cv2.imwrite(filename, image)


def main(unused_argv):
    video_to_img(FLAGS.base_dir, FLAGS.output_dir)



if __name__ == '__main__':
    tf.app.run()
