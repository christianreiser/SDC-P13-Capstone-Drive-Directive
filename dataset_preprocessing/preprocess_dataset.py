import argparse
import os
import glob
import sys
import numpy as np
from PIL import Image

# Parse arguments from terminal
parser = argparse.ArgumentParser()

parser.add_argument('--read_path', '-r', type=str, required=True,
                    help='Path to directory containing original images')

parser.add_argument('--write_directory', '-w', type=str, required=True,
                    help='Name of directory where processed images will be saved')

parser.add_argument('--class_names', '-c', nargs='+', type=str, required=True,
                    help='Class names corresponding to original image file base names')

parser.add_argument('--num_examples', '-n', type=int, required=True,
                    help='Number of examples to keep from each class')

parser.add_argument('--img_width', '-i', type=int, required=True,
                    help='Processed image width (aspect ratio maintained)')

args = parser.parse_args()


def preprocess(read_path, write_path, class_name, num_examples, img_width):

    # Create an array of image file paths
    img_path_array = np.array(glob.glob(os.path.join(read_path, class_name + '*.jpg')))

    # Shuffle images file paths
    np.random.shuffle(img_path_array)

    # Reduce number of images file paths
    img_array_reduced = img_path_array[:num_examples]

    # Read, resize, and write images to new path
    img = Image.open(img_array_reduced[0])
    resize_percent = img_width / float(img.size[0])
    height = int(float(img.size[1]) * resize_percent)

    for i in range(len(img_array_reduced)):

        img = Image.open(img_array_reduced[i])
        img = img.resize((img_width, height))
        img_write_path = os.path.join(write_path, class_name + '{:0>4}'.format(i) + '.jpg')
        img.save(img_write_path)

    return None


def main():

    write_path = os.path.join(os.path.dirname(args.read_path), args.write_directory)

    # Check if directory exists
    if os.path.exists(write_path):
        sys.exit('Write directory exists. Exiting.')
    else:
        os.makedirs(write_path)

    for class_name in np.array(args.class_names):

        preprocess(args.read_path, write_path, class_name, args.num_examples, args.img_width)


if __name__ == '__main__':
    main()
