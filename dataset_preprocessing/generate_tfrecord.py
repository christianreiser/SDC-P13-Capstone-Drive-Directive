import os
import subprocess
import sys
import argparse
import io

import tensorflow as tf
import pandas as pd

from collections import namedtuple
from PIL import Image

parent_path = os.path.dirname(os.getcwd())
models_path = os.path.join(parent_path, 'models')

# Check if tensorflow/models repo exists in parent directory
if not os.path.exists(models_path):

    # Clone tensorflow/models repo into parent directory if does not exist
    tf_models_repo = 'https://github.com/tensorflow/models.git'
    subprocess.call(['git', 'clone', tf_models_repo, models_path])

# Add path to dataset_util
sys.path.insert(0, os.path.join(parent_path, 'models/research/object_detection/utils'))

import dataset_util

# Parse arguments from terminal
parser = argparse.ArgumentParser()
parser.add_argument('--read_path', '-r', type=str, required=True,
                    help='Path to directory containing dataset images')
parser.add_argument('--labels_filename', '-l', type=str, required=True,
                    help='Name of file containing dataset labels')
parser.add_argument('--save_filename', '-s', type=str, required=True,
                    help='Name of file to be saved')
args = parser.parse_args()


def create_tf_example(example, directory_path):

    filename = example.image.encode('utf8')

    # Encode image
    with tf.gfile.GFile(os.path.join(directory_path, example.image), 'rb') as fid:
        encoded_image_data = fid.read()

    encoded_image_data_io = io.BytesIO(encoded_image_data)
    img = Image.open(encoded_image_data_io)
    width, height = img.size
    image_format = 'jpg'.encode('utf8')

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    for index, row in example.object.iterrows():

        xmins.append(float(row['xMin']) / width)
        xmaxs.append(float(row['xMax']) / width)
        ymins.append(float(row['yMin']) / height)
        ymaxs.append(float(row['yMax']) / height)
        classes_text.append(row['name'].encode('utf8'))

        if row['name'] == 'red':
            classes.append(1)
        elif row['name'] == 'yellow':
            classes.append(2)
        elif row['name'] == 'green':
            classes.append(3)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def main():

    labels_path = os.path.join(args.read_path, args.labels_filename)
    record_path = os.path.join(args.read_path, args.save_filename)

    # Create TFRecord writer
    writer = tf.python_io.TFRecordWriter(record_path)

    # Read in labels file
    examples = pd.read_csv(labels_path)

    data = namedtuple('data', ['image', 'object'])
    files_grouped = examples.groupby('image')

    # Create an array with annotation box coordinates grouped by image
    examples_grouped = [data(image, files_grouped.get_group(x)) for image, x in
                        zip(files_grouped.groups.keys(), files_grouped.groups)]

    for example in examples_grouped:

        # Skip labels file header
        if example[0] != 'image':
            tf_example = create_tf_example(example, args.read_path)
            writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    main()
