import tensorflow as tf
from generator import Generator
import random
import numpy as np 

class Dataset(object):
    sample_resized = 'sample_resized'
    label = 'label'

    # def __init__(self, generator=Generator()):
    def __init__(self, generator, batch_size=32, prefetch_batch_buffer=2, target_size=[224, 224], getLink=False):
        #get link for debug only
        self.getLink = getLink
        self.target_size=target_size
        self.next_element = self.build_iterator(generator, batch_size, prefetch_batch_buffer)

    def build_iterator(self, gen: Generator, batch_size, prefetch_batch_buffer):

        dataset = tf.data.Dataset.from_generator(gen.get_next,
                                                 output_types={Generator.sample: tf.string,
                                                               Generator.label: tf.float32,
                                                               Generator.augment: tf.bool})
        dataset = dataset.map(self._read_image_and_augment)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        # return Inputs(element[self.img1_resized],
        #               element[self.img2_resized],
        #               element[PairGenerator.label])
        if self.getLink: 
            return element[self.sample_resized], element[Generator.label], element[Generator.sample] 
        else:
            return element[self.sample_resized], element[Generator.label]# link
    
    def flip(self,img):
        type_flip = tf.random_uniform(shape=[1], minval=0, maxval=2, dtype=tf.int32) # random 0,1
        img = tf.cond(
            tf.equal(type_flip[0], 0),
            lambda: tf.image.flip_up_down(img), # true
            lambda: tf.image.flip_left_right(img), # FALSE
        )
        return img

    def rotate(self,img):
        angle = tf.random_uniform(shape=[1], minval=0, maxval=2*3.14)
        return tf.contrib.image.rotate(img,angles=angle[0])

    def translate(self,img):
        return tf.contrib.image.translate(img, tf.random_uniform(shape=[2], minval=0, maxval=15))

    def augment(self, img): 
        flip = lambda: self.flip(img) # case 1
        rotate = lambda: self.rotate(img) # case 2
        translate = lambda: self.translate(img) # case 3
        not_aug = lambda: img # case 0 and default
        type_aug = tf.random_uniform(shape=[1], minval=0, maxval=4, dtype=tf.int32) # random 0,1,2,3
        img_augmented = tf.case(
            {tf.equal(type_aug[0], 0): not_aug, tf.equal(type_aug[0], 1): flip, tf.equal(type_aug[0], 2): rotate, tf.equal(type_aug[0], 3): translate},
            default=not_aug, 
            exclusive=True)
        type_aug = random.choice([0,1,2,3])
        if type_aug == 1: # flip
            type_flip = random.choice(['up_down', 'left_right'])
            if type_flip == 'up_down':
                img = tf.image.flip_up_down(img)
            else:
                img = tf.image.flip_left_right(img)

        elif type_aug == 2: # rotate
            img = tf.contrib.image.rotate(img,angles=tf.random_uniform(shape=[1], minval=0, maxval=2*3.14))
        
        elif type_aug == 3: # translate
            img = tf.contrib.image.translate(
                img,
                tf.random_uniform(shape=[2], minval=0, maxval=15),
            )
        else:
            print('train : not use augment')
            
        return img_augmented

    def not_augment(self, img): 
        print('val: not use augment')
        return img

    def _read_image_and_augment(self, element):

        print('>>>>>>>>. element: ', element)
        # element la` 1 dict do generator.get_next trả về, gồm link ảnh và label
        target_size = self.target_size # 224, 224
        # read images from disk
        img_file = tf.read_file(element[Generator.sample]) # read image file from generator output
        img = tf.image.decode_image(img_file)
        img = img[100:350,100:350,:]
        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img.set_shape([None, None, 3])
        # resize to model input size
        img = tf.image.resize_images(img, target_size)

        # =============== add augmentaion ============

        # img = tf.cond(
        #     element[Generator.augment],
        #     lambda: self.augment(img),
        #     lambda: self.not_augment(img),
        # )
        # =============== add augmentaion ============

        element[self.sample_resized] = img/255.0 # normalize

        # element[self.label] = tf.cast(element[PairGenerator.label], tf.float32)

        return element
