import numpy as np
import cv2
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))

def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}



def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, 0, 0]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return np.asarray(estimated_boxes), len(contours)
