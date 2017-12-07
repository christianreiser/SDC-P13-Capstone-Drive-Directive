# Converts the yaml definition files from Bosch Small Traffic Lights Dataset
# to the format used in this project
# see https://hci.iwr.uni-heidelberg.de/node/6132
#
# Output format:
# filename,width,height,class,xmin,ymin,xmax,ymax
# red0031.jpg,300,240,red,123,81,134,109
# red0025.jpg,300,240,red,110,81,121,108

import yaml
import csv
from os import listdir
from os.path import isfile, join

yaml_path = '../data/yaml/'
csv_path = '../data/csv/'
width = 1280
height = 720


def convert_box(path, box):
    label = 'non-red'
    if box['label'] == 'Red':
        label = 'red'
    return [path, label, width, height, int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max'])]


def convert_file(yaml_file, csv_file):
    labels = [['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]

    with open(yaml_file, 'r') as stream:
        try:
            document = yaml.load(stream)
            for element in document:
                boxes = element['boxes']
                path = element['path']
                for box in boxes:
                    label = convert_box(path, box)
                    labels.append(label)

        except yaml.YAMLError as exc:
            print(exc)

    print 'Found', len(labels) - 1, 'labels'

    with open(csv_file, 'w') as stream:
        writer = csv.writer(stream)
        writer.writerows(labels)


def convert_folder(yaml_folder=yaml_path, csv_folder=csv_path):
    yaml_files = [f for f in listdir(yaml_folder) if isfile(join(yaml_folder, f)) and f.endswith('.yaml')]

    for f in yaml_files:
        print 'Parsing', f, '...'

        yaml_file = join(yaml_folder, f)
        csv_file = join(csv_folder, f.replace('.yaml', '.csv'))

        convert_file(yaml_file, csv_file)


if __name__ == "__main__":
    convert_folder(yaml_folder=yaml_path, csv_folder=csv_path)
