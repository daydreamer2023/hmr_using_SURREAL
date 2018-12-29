from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, exists
from os import makedirs, listdir, path
from os import makedirs

import numpy as np
import imageio
import cv2

import tensorflow as tf

tf.app.flags.DEFINE_string('base_dir',
                           '/home/cha/SamSung/DataSet/data/SURREAL',
                           'base dir')

FLAGS = tf.app.flags.FLAGS


def vid_to__img(base_dir, split):
    base_dir = join(base_dir, split)

    for f1 in sorted(listdir(base_dir)):
        for f2 in sorted(listdir(join(base_dir, f1))):
            print(join(base_dir, f1, f2))
            for f3 in sorted(listdir(join(base_dir, f1, f2))):
                if f3.endswith('.mp4'):
                    vid_name = join(base_dir, f1, f2, f3)
                    vid = imageio.get_reader(vid_name, 'ffmpeg')

                    vid_parse = path.splitext(path.basename(f3))[0]
                    img_dir = join(base_dir, f1, f2, vid_parse)

                    if not exists(img_dir):
                        makedirs(img_dir)

                    for frame, im in enumerate(vid):
                        image = vid.get_data(frame)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        filename = join(img_dir, 'frame_%d.jpg' % (frame + 1))
                        cv2.imwrite(filename, image)

def main(unused_argv):
    vid_to__img(FLAGS.base_dir, 'test')
    vid_to__img(FLAGS.base_dir, 'train')
    vid_to__img(FLAGS.base_dir, 'val')

if __name__ == '__main__':
    tf.app.run()