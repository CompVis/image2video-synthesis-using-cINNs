import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
import imageio

## Code adapted from https://github.com/edenton/svg/blob/master/data/convert_bair.py

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='base directory where tf records files are placed')
parser.add_argument('--output_dir', required=True, help='output directory to save processed data')
opt = parser.parse_args()


def get_seq(dname):
    data_dir = '%s/softmotion30_44k/%s' % (opt.data_dir, dname)
    # data_dir = '%s/%s' % (opt.data_dir, dname)


    filenames = gfile.Glob(os.path.join(data_dir, '*tfrecords'))
    if not filenames:
        raise RuntimeError('No data files found.')

    for f in filenames:
        k=0
        for serialized_example in tf.compat.v1.io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            endeffector_positions = np.empty((0, 3), dtype='float')

            for i in range(30):
                image_name = str(i) + '/image_aux1/encoded'
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                img = Image.frombytes('RGB', (64, 64), byte_str)
                arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                image_seq.append(arr.reshape(1, 64, 64, 3))

                endeffector_pos_name = str(i) + '/endeffector_pos'
                endeffector_pos_value = list(example.features.feature[endeffector_pos_name].float_list.value)
                endeffector_positions = np.vstack((endeffector_positions, endeffector_pos_value))

            image_seq = np.concatenate(image_seq, axis=0)
            k=k+1
            yield f, k, image_seq, endeffector_positions


def convert_data(dname):
    seq_generator = get_seq(dname)
    n = 0
    while True:
        n+=1
        try:
            f, k, seq, pos = next(seq_generator)
        except StopIteration:
            break
        f = f.split('/')[-1]
        output_dir = opt.output_dir
        os.makedirs('%s/%s/%s/%d/' % (output_dir, dname, f[:-10], k), exist_ok=True)
        np.savetxt('/%s/%s/%s/%d/endeffector_positions.csv' % (output_dir, dname, f[:-10], k), pos, delimiter=',')
        for i in range(len(seq)):
            imageio.imwrite('/%s/%s/%s/%d/%d.png' % (output_dir, dname, f[:-10], k, i), seq[i].astype(np.uint8))

        print('%s data: %s (%d)  (%d)' % (dname, f, k, n))


if __name__ == '__main__':
    import datetime
    import os
    import time
    print('Program started at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    start_time = time.time()
    convert_data('train')
    # convert_data('test')
    end_time = time.time()
    print('Program ended at ' + datetime.datetime.now().strftime('%d/%m/%Y %I:%M:%S %p'))
    print('Execution time: ' + str(datetime.timedelta(seconds=end_time - start_time)))

#