import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import pp, visualize, to_json, show_all_variables
from models import ALOCC_Model
import matplotlib.pyplot as plt
from kh_tools import *
import numpy as np
import scipy.misc
from utils import *
import time
import os
from skimage.util import random_noise
import numpy as np
import time

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("attention_label", 1, "Conditioned label that growth attention of training label [1]")
flags.DEFINE_float("r_alpha", 0.2, "Refinement parameter [0.2]")
flags.DEFINE_integer("train_size", 10000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 45, "The size of image to use. [45]")
flags.DEFINE_integer("input_width", None, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 45, "The size of the output images to produce [45]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string("dataset_address", "./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test", "The path of dataset")
flags.DEFINE_string("input_fname_pattern", "*", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/UCSD_128_45_45/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")

FLAGS = flags.FLAGS
processed = []

def get_latest_image(dirpath, valid_extensions=('jpg','jpeg','png')):
    """
    Get the latest image file in the given directory
    """
    global processed
    f = True
    valid_files = [os.path.join(dirpath, filename) for filename in os.listdir(dirpath)]
    new_files = [z for z in valid_files if not z in processed]
    processed.extend(new_files)
    #print(new_files,'\n',processed)
    return new_files
    '''valid_files = [f for f in valid_files if '.' in f and f.rsplit('.',1)[-1] in valid_extensions and os.path.isfile(f)]
    if not valid_files:
        f = True
    else:
        f = False
    return max(valid_files, key=os.path.getmtime)'''


def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

def main(_):
    print('Program is started at', time.clock())
    pp.pprint(flags.FLAGS.__flags)

    n_per_itr_print_results = 100
    n_fetch_data = 1
    kb_work_on_patch= False
    nd_input_frame_size = (200, 360)
    nd_patch_size = (315, 180)
    n_stride = 64
    FLAGS.checkpoint_dir = "./checkpoint/my_data_6_180_315/"

    FLAGS.dataset = 'my_data'
    FLAGS.dataset_address = './test_data'
    lst_test_dirs = ['Test']

    FLAGS.input_width = nd_patch_size[0]
    FLAGS.input_height = nd_patch_size[1]
    FLAGS.output_width = nd_patch_size[0]
    FLAGS.output_height = nd_patch_size[1]


    check_some_assertions()

    nd_patch_size = (FLAGS.input_width, FLAGS.input_height)
    nd_patch_step = (n_stride, n_stride)
    FLAGS.train = False
    FLAGS.epoch = 1
    FLAGS.batch_size = 1


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    run_config = tf.ConfigProto(gpu_options=gpu_options)
    run_config.gpu_options.allow_growth=True
    with tf.Session(config=run_config) as sess:
        tmp_ALOCC_model = ALOCC_Model(
                    sess,
                    input_width=FLAGS.input_width,
                    input_height=FLAGS.input_height,
                    output_width=FLAGS.output_width,
                    output_height=FLAGS.output_height,
                    batch_size=FLAGS.batch_size,
                    sample_num=FLAGS.batch_size,
                    attention_label=FLAGS.attention_label,
                    r_alpha=FLAGS.r_alpha,
                    is_training=FLAGS.train,
                    dataset_name=FLAGS.dataset,
                    dataset_address=FLAGS.dataset_address,
                    input_fname_pattern=FLAGS.input_fname_pattern,
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    sample_dir=FLAGS.sample_dir,
                    nd_patch_size=nd_patch_size,
                    n_stride=n_stride,
                    n_per_itr_print_results=n_per_itr_print_results,
                    kb_work_on_patch=kb_work_on_patch,
                    nd_input_frame_size = nd_input_frame_size,
                    n_fetch_data=n_fetch_data)

        show_all_variables()

        print('--------------------------------------------------')
        print('Load Pretrained Model...')
        tmp_ALOCC_model.f_check_checkpoint()

        '''for s_image_dirs in sorted(glob(os.path.join(FLAGS.dataset_address,'Test_Unreal'))):
            tmp_lst_image_paths = []
            
            for img_file in sorted(glob(os.path.join(s_image_dirs + '/*'))):
                #lst_image_paths = [img_file] #tmp_lst_image_paths
                tmp_image = read_image(img_file)
                sigma = 0.155
                noisy = random_noise(tmp_image, var=sigma ** 2)
                #images = read_lst_images_w_noise2(lst_image_paths, nd_patch_size, nd_patch_step)
                image = np.array(noisy)

                lst_prob = process_frame(image,tmp_ALOCC_model,cnt)
                cnt += 1
                print('test for img {} is finished'.format(img_file))'''
        chk = True
        cnt = 0
        while chk:
            img_files = get_latest_image('/home/deep/catkin_ws/src/detector/src/img_folder')
            if not img_files:
                continue
            
            time.sleep(1)

            for img_file in img_files:
                tmp_image = read_image(img_file)
                sigma = 0.1
                noisy = random_noise(tmp_image, var=sigma ** 2)
                #images = read_lst_images_w_noise2(lst_image_paths, nd_patch_size, nd_patch_step)
                image = np.array(noisy)

                discri_val = process_frame(image,tmp_ALOCC_model,cnt)

                if discri_val >= 10.0:
                    print('discriminator value = {}'.format(discri_val))
                    scipy.misc.imsave('/home/deep/Desktop/anomaly/anomaly_{}.jpg'.format(cnt), tmp_image)
                    cnt += 1



def process_frame(frames_src,sess,img_file):
    #nd_patch,nd_location = get_image_patches(frames_src,sess.patch_size,sess.patch_step)
    #frame_patches = nd_patch#.transpose([1,0,2,3])
    discri_val = sess.f_test_frozen_model(frames_src,img_file)
    return discri_val


if __name__ == '__main__':
    tf.app.run()


