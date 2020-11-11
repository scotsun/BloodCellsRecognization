import numpy as np
from xml.etree.ElementTree import parse


class XmlParser(object):
    def __init__(self):
        self.tree = None

    def _tree(self, annotation_file):
        self.tree = parse(annotation_file)
        return

    def _root_tag(self, annotation_file):
        self._tree(annotation_file)
        root = self.tree.getroot()
        return root

    def get_fname(self, annotation_file):
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        self._tree(annotation_file)
        for element in self.tree.iter():
            if 'width' in element.tag:
                return float(element.text)

    def get_height(self, annotation_file):
        self._tree(annotation_file)
        for element in self.tree.iter():
            if 'height' in element.tag:
                return float(element.text)

    def get_labels(self, annotation_file):
        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels

    def get_boxes(self, annotation_file):
        root = self._root_tag(annotation_file)
        bounding_boxes = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([float(x1), float(x2), float(y1), float(y2)])
            bounding_boxes.append(box)
        bounding_boxes = np.array(bounding_boxes)
        return bounding_boxes

    def xmls_to_csv(self):
        pass