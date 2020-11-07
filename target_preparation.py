import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def get_raw_data_from_annotation(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'cell_type', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    for i in range(len(xml_df['filename'])):
        if '.jpg' not in xml_df.filename[i]:
            xml_df.filename[i] = xml_df.filename[i] + '.jpg'
    return xml_df


def get_parse(filepath):
    # TODO:
    filename = ""
    labels = []
    boxes = []
    return filename, labels, boxes


def generator(batch_size, num_classes, annotation_folder):
    pattern_shape = [52, 26, 13]
    anchor_shape = [3, 3]
    while True:
        input = []
        ytrue = [np.zeros((batch_size, pattern_shape[l], pattern_shape[l], 3, 5 + len(num_classes)))
                 for l in range(anchor_shape[0])]

        for i in range(batch_size):
            filename, labels, boxes = get_parse(annotation_folder[i])

        yield input, ytrue

    pass
