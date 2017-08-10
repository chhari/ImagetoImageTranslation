# Writtem by Weiwei Tao
#
#Reference:     https://github.com/affinelayer/pix2pix-tensorflow
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math

CROP_SIZE = 256

def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1

def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]

def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(srgb, [-1, 3])
    #RGB cannot be converted to LAB directly
	#RGB to XYZ
    with tf.name_scope("srgb_to_xyz"):
        linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
        exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
        rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
        xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)
     
	#XYZ TO LAB
    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    with tf.name_scope("xyz_to_cielab"):
        # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)
        # normalize for D65 white point
        xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])
        
        epsilon = 6/29
        linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
        exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
        fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask
        # convert to lab
        fxfyfz_to_lab = tf.constant([
            #  l       a       b
            [  0.0,  500.0,    0.0], # fx
            [116.0, -500.0,  200.0], # fy
            [  0.0,    0.0, -200.0], # fz
            ])
        lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])
    return tf.reshape(lab_pixels, tf.shape(srgb))


	
def load_examples( lab_colorization=True, which_direction="AtoB", flip=True, scale_size=286, batch_size=1):
    """Reads directory of images in tensorflow
        Args:
            input_dir: the directory of input images
            lab_colorization:
            which_direction:
            flip: default true
            scale_size: default 286
            batch_size: default 1
        Returns:ss
    """
    ##should change input_dir 
    input_dir = './facade/test'
    
    #input images should either be .jpg or .png or .jpeg
    input_paths = glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]'))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, '*.[jJ][pP][eE][gG]'))
        decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]'))
        decode = tf.image.decode_png

    #Sort images by name
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    # if the image names are numbers, sort by the value 
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        #built in function, return a new sorted list
    else:
        input_paths = sorted(input_paths)
    print('1- input_paths shape is ', len(input_paths))
    #read, decode and post process images
    with tf.name_scope("load_images"):
        # Create a queue node (or operation) in the currently active graph
        path_queue = tf.train.string_input_producer(input_paths)
		
        #Read the files
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        
        #Decode the images
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        
        print('Before assertion the last dimension is ',list(raw_input.get_shape())[2])
        
		#make the last dimension 3 so that we can unstack the colors
        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)
        
        raw_input.set_shape([None, None, 3])

        print('last dimension is ',list(raw_input.get_shape())[2])
        
        if lab_colorization:
            # load color and brightness from image, no B image exists here
			#convert RGB information to LAB 
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
			print('last dimension of L_chan is ',list(L_chan.get_shape()))
			#expand_dim won't add dimension to a tensor. 
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
			#left half and right side
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])
    
    if which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")


    # !!IMPORTANT: seed for image operations to do the SAME operations to both input and output images
    seed = random.randint(0, 2**31 - 1)

    def transform(image):
        r = image
        if flip:
            r = tf.image.random_flip_left_right(r, seed=seed)
        
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r
	
    #distort the input images
    with tf.name_scope("input_images"):
        input_images = transform(inputs)
    #distort the target images
    with tf.name_scope("target_images"):
        target_images = transform(targets)
    
    
    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size = batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) /batch_size))
    
    print('2- count is ',len(input_paths))
    print('3- steps_per_epoch is ',steps_per_epoch)

    return paths_batch, inputs_batch, targets_batch, len(input_paths), steps_per_epoch


def main():
    paths, inputs, targets, count, steps_per_epoch = load_examples()
    #print("paths: {}, inputs: {} examples count = {}" %(paths, inputs, count))


main()

