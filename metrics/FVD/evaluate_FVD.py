from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, torch
from tqdm import tqdm
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow.compat.v1 as tf
from metrics.FVD.FVD import Embedder, preprocess, calculate_fvd
import numpy as np


def compute_fvd(real_videos, fake_videos, device=0):
    devs = tf.config.experimental.get_visible_devices("GPU")
    target_dev = [d for d in devs if d.name.endswith(str(device))][0]
    tf.config.experimental.set_visible_devices(target_dev, 'GPU')

    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            # construct graph
            sess = tf.Session()
            input_real = tf.placeholder(dtype=tf.float32, shape=(*real_videos[0].shape[:2], real_videos[0].shape[3],
                                                                 real_videos[0].shape[4], real_videos[0].shape[2]))
            input_fake = tf.placeholder(dtype=tf.float32, shape=(*real_videos[0].shape[:2], real_videos[0].shape[3],
                                                                 real_videos[0].shape[4], real_videos[0].shape[2]))

            real_pre = preprocess(input_real, (224, 224))
            emb_real = Embedder(real_pre)
            embed_real = emb_real.create_id3_embedding(real_pre)
            fake_pre = preprocess(input_fake, (224, 224))
            emb_fake = Embedder(fake_pre)
            embed_fake = emb_fake.create_id3_embedding(fake_pre)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            real, fake = [], []
            for rv, fv in tqdm(zip(real_videos, fake_videos)):
                real_batch = ((rv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                fake_batch = ((fv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                feed_dict = {input_real: real_batch, input_fake: fake_batch}
                r, f = sess.run([embed_fake, embed_real], feed_dict)
                real.append(r); fake.append(f)

            print('Compute FVD score')
            real = np.concatenate(real, axis=0)
            fake = np.concatenate(fake, axis=0)
            embed_real = tf.placeholder(dtype=tf.float32, shape=(real.shape[0], 400))
            embed_fake = tf.placeholder(dtype=tf.float32, shape=(real.shape[0], 400))
            result = calculate_fvd(embed_real, embed_fake)
            feed_dict = {embed_real: real, embed_fake: fake}
            result = sess.run(result, feed_dict)
            sess.close()
    tf.reset_default_graph()
    return result

def get_embeddings(fake_videos, device=0):

    devs = tf.config.experimental.get_visible_devices("GPU")
    target_dev = [d for d in devs if d.name.endswith(str(device))][0]
    tf.config.experimental.set_visible_devices(target_dev, 'GPU')

    with tf.device("/gpu:0"):
        with tf.Graph().as_default():
            # construct graph
            sess = tf.Session()
            input_fake = tf.placeholder(dtype=tf.float32, shape=(*fake_videos[0].shape[:2], fake_videos[0].shape[3],
                                                                 fake_videos[0].shape[4], fake_videos[0].shape[2]))

            fake_pre = preprocess(input_fake, (224, 224))
            emb_fake = Embedder(fake_pre)
            embed_fake = emb_fake.create_id3_embedding(fake_pre)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            real, fake = [], []
            for fv in tqdm(fake_videos):
                fake_batch = ((fv + 1.) * 127.5).permute(0, 1, 3, 4, 2).cpu().numpy()
                feed_dict = {input_fake: fake_batch}
                f = sess.run([embed_fake], feed_dict)
                fake.append(f)

            fake = np.concatenate(fake, axis=0)
            sess.close()
    tf.reset_default_graph()
    return fake