import glob
import os
import numpy as np
from PIL import Image
from xml_parser import XmlParser


def get_parse(ann_fname, input_size):
    parser = XmlParser()
    WIDTH = parser.get_width(ann_fname)
    HEIGHT = parser.get_height(ann_fname)
    filename = parser.get_fname(ann_fname)
    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for i in range(len(boxes)):
        boxes[i][0] = boxes[i][0] / WIDTH * input_size
        boxes[i][1] = boxes[i][1] / WIDTH * input_size
        boxes[i][2] = boxes[i][2] / HEIGHT * input_size
        boxes[i][3] = boxes[i][3] / HEIGHT * input_size

    return filename, labels, boxes


def get_iou(box1, box2):
    w_min = min(box1[1], box2[1])
    h_min = min(box1[3], box2[3])
    w = w_min - box1[0]
    h = h_min - box1[2]
    intersect = w * h
    merge = (box1[1] - box1[0]) * (box1[3] - box1[2]) + (box2[1] - box2[0]) * (box2[3] - box2[2])
    iou = intersect / (merge - intersect)
    return iou


def get_anchor(anchors, box):
    iou_list = []
    anchors_list = np.zeros((len(anchors), 4), dtype='float32')  # list of anchors used to fit the box
    for i in range(len(anchors_list)):
        anchors_list[i][0] = box[0]
        anchors_list[i][1] = anchors_list[i][0] + anchors[i][0]
        anchors_list[i][2] = box[2]
        anchors_list[i][3] = anchors_list[i][2] + anchors[i][1]

        iou = get_iou(box, anchors_list[i])
        iou_list.append(iou)

    anchor_ix = np.argmax(iou_list)
    return anchor_ix


def get_img(img_dir, fname, input_size) :
    img_fname = os.path.join(img_dir, fname)
    image = Image.open(img_fname)
    img = np.array(image.resize((input_size, input_size)))
    return img


def get_yture(boxes, anchors, anchor_shape, b, pattern_shape, input_size, classes, labels, ytrue):
    newbox = np.zeros(4, dtype='float32')
    for i in range(len(boxes)):
        anchor_ix = get_anchor(anchors, boxes[i])

        layer_anchor = anchor_ix // anchor_shape[1]
        box_anchor = anchor_ix % anchor_shape[1]

        rate = pattern_shape[layer_anchor] / input_size

        center_x = (boxes[i][0] + boxes[i][1]) / 2 * rate
        center_y = (boxes[i][2] + boxes[i][3]) / 2 * rate

        x = np.floor(center_x).astype('int32')
        y = np.floor(center_y).astype('int32')
        w = boxes[i][1] - boxes[i][0]
        h = boxes[i][3] - boxes[i][2]

        c = classes.index(labels[i])

        newbox[0] = center_x - x
        newbox[1] = center_y - y
        newbox[2] = np.log(w / anchors[anchor_ix][0])
        newbox[3] = np.log(h / anchors[anchor_ix][1])

        ytrue[layer_anchor][b, x, y, box_anchor, 0:4] = newbox[0:4]
        ytrue[layer_anchor][b, x, y, box_anchor, 4] = 1
        ytrue[layer_anchor][b, x, y, box_anchor, 5 + c] = 1

    return ytrue


def generator(batch_size, classes, ann_fnames, input_size, anchors, img_dir):
    pattern_shape = [52, 26, 13]
    anchor_shape = [3, 3]
    n = len(ann_fnames)
    i = 0
    while True:
        input = []
        ytrue = [np.zeros((batch_size, pattern_shape[l], pattern_shape[l], 3, 5 + len(classes)))
                 for l in range(anchor_shape[0])]

        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(ann_fnames)
            fname, labels, boxes = get_parse(ann_fnames[i], input_size)
            ytrue = get_yture(boxes, anchors, anchor_shape, b, pattern_shape, input_size, classes, labels, ytrue)
            img = get_img(img_dir, fname, input_size)
            input.append(img)
            i = (i + 1) % n
        input = np.array(input)
        yield input, [ytrue[2], ytrue[1], ytrue[0]]


# root = os.path.dirname(__file__)
# ann_dir = os.path.join(root, "data", "ann", "*.xml")
# ann_fnames = glob.glob((ann_dir))
# img_dir = os.path.join(root, "data", "img")