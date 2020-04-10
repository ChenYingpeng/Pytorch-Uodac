# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import time
import csv

classname_to_id = {'holothurian':0,'echinus':1,'scallop':2,'starfish':3}
class_ids = []
for key, value in classname_to_id.items():
    class_ids.append(value)


def nms(dets, iou_thr=0.5):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]        # clw note: for x1y1x2y2
    y2 = dets[:, 3]
    # x2 = dets[:, 2] + x1     # clw modify: for xywh
    # y2 = dets[:, 3] + y1
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def soft_nms(dets, sigma=0.5, iou_thr=0.5, method='linear', min_score=0.0001):
    '''
    Arguments:
        dets (np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for soft_nms.
        method: 'linear' or 'gaussian'
        min_score: if box's score lower than min_score，the box will be deprecated by exchanging with the last box.
    '''
    box_len = len(dets)   # box的个数
    for i in range(box_len):
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]
        max_pos = i
        max_scores = ts

        # get max box
        pos = i+1
        while pos < box_len:
            if max_scores < dets[pos, 4]:
                max_scores = dets[pos, 4]
                max_pos = pos
            pos += 1

        # add max box as a detection
        dets[i, :] = dets[max_pos, :]

        # swap ith box with position of max box
        dets[max_pos, 0] = tmpx1
        dets[max_pos, 1] = tmpy1
        dets[max_pos, 2] = tmpx2
        dets[max_pos, 3] = tmpy2
        dets[max_pos, 4] = ts

        # 将置信度最高的 box 赋给临时变量
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i, 0], dets[i, 1], dets[i, 2], dets[i, 3], dets[i, 4]

        pos = i+1
        # NMS iterations, note that box_len changes if detection boxes fall below threshold
        while pos < box_len:
            x1, y1, x2, y2 = dets[pos, 0], dets[pos, 1], dets[pos, 2], dets[pos, 3]

            area = (x2 - x1 + 1)*(y2 - y1 + 1)

            iw = (min(tmpx2, x2) - max(tmpx1, x1) + 1)
            ih = (min(tmpy2, y2) - max(tmpy1, y1) + 1)
            if iw > 0 and ih > 0:
                overlaps = iw * ih
                ious = overlaps / ((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1) + area - overlaps)

                if method =='linear':    # 线性
                    if ious > iou_thr:
                        weight = 1 - ious
                    else:
                        weight = 1
                elif method == 'gaussian':  # gaussian
                    weight = np.exp(-(ious**2) / sigma)
                else:
                    raise ValueError('Invalid method for SoftNMS: {}'.format(method))
                    # original NMS
                    # if ious > thresh:
                    #     weight = 0
                    # else:
                    #     weight = 1

                # 赋予该box新的置信度
                dets[pos, 4] = weight * dets[pos, 4]

                # 如果box得分低于阈值thresh，则通过与最后一个框交换来丢弃该框
                if dets[pos, 4] < min_score:
                    dets[pos, 0] = dets[box_len-1, 0]
                    dets[pos, 1] = dets[box_len-1, 1]
                    dets[pos, 2] = dets[box_len-1, 2]
                    dets[pos, 3] = dets[box_len-1, 3]
                    dets[pos, 4] = dets[box_len-1, 4]

                    box_len = box_len-1
                    pos = pos-1
            pos += 1

    keep = [i for i in range(box_len)]
    return keep


def box_iou_vote(b1, b2):

    b1 = np.expand_dims(b1, -2)
    b2 = np.expand_dims(b2, 0)

    b1_mins = b1[..., :2]
    b1_maxes = b1[..., 2:4]
    b1_wh = b1[..., 2:4] - b1[..., 0:2]        # clw note: for x1y1x2y2
    # b1_maxes = b1[..., :2] + b1[..., 2:4]
    # b1_wh = b1[..., 2:4]                     # for xywh

    b2_mins = b2[..., :2]
    b2_maxes = b2[..., 2:4]
    b2_wh = b2[..., 2:4] - b2[..., 0:2]        #
    # b2_maxes = b2[..., :2] + b2[..., 2:4]
    # b2_wh = b2[..., 2:4]                     # for xywh

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_voting(boxes, boxes_all, th_vote):
    mask_iou = box_iou_vote(boxes, boxes_all)  # clw note: add mode='xyxy' or 'xywh'  TODO
    mask_iou = mask_iou >= th_vote

    for i, box in enumerate(boxes):
        boxes_sample = boxes_all[mask_iou[i]]
        boxes_sample = boxes_sample[:, :4] * boxes_sample[:, -1:] / np.sum(boxes_sample[:, -1:])
        boxes[i, :4] = np.sum(boxes_sample[:, :4], axis=0)

    return boxes

def get_img_names_in_csv(csv_path):
    csv_reader = csv.reader(open(csv_path, 'r'))
    img_names = []
    for i, box in enumerate(csv_reader):
        if i == 0:
            continue
        else:
            if box[1][:-4] + '.jpg' not in img_names:
                img_names.append(box[1][:-4] + '.jpg')
            else:
                continue
    return img_names


def merger_csv():

    csv1_path = 'submit/test_A_image_submission_htc_dconv_c3-c5_mstrain_600_1000_x101_64x4d_fpn_12e_ac_aug.csv'
    csv2_path = 'submit/test_A_image_submission_htc_dconv_c3-c5_mstrain_600_1000_x101_64x4d_fpn_12e_aug_049920882.csv'
    timestamp = time.strftime('_%Y%m%d_%H%M%S', time.localtime())  # clw modify
    json_output = 'results/x101+r101' + str(timestamp) + '.json'
    csv_output = 'submit/x101+r101' + str(timestamp) + '.csv'
    #test_path = 'D:/2020water/test-A-image'

    csv_reader_1 = csv.reader(open(csv1_path, 'r'))
    csv_reader_2 = csv.reader(open(csv2_path, 'r'))

    #img_list = os.listdir(test_path)
    img_list = get_img_names_in_csv(csv2_path)

    th_nms = 0.5
    vote = False  #
    th_vote = 0.5

    result_json = []                                                 # clw note: for json
    result_csv = ["name,image_id,confidence,xmin,ymin,xmax,ymax\n"]  # clw note: for csv

    start = time.time()
    csv_data_1 = []
    for item in csv_reader_1:
        csv_data_1.append(item)
    csv_data_2 = []
    for item in csv_reader_2:
        csv_data_2.append(item)
    print('read file time use: %.3fs' % (time.time()-start))
    for i, img_name in enumerate(img_list):
        print(i, img_name)
        boxes = []

        for box in csv_data_1:
            img_name_1 = box[1][:-4] + '.jpg'
            if img_name != img_name_1:
                continue
            else:
                category = classname_to_id[box[0]]
                bbox = [float(box[3]), float(box[4]), float(box[5]), float(box[6])]
                score = float(box[2])
                boxes.append(bbox + [score] + [category])
        for box in csv_data_2:
            img_name_2 = box[1][:-4] + '.jpg'
            if img_name != img_name_2:
                continue
            else:
                category = classname_to_id[box[0]]
                bbox = [float(box[3]), float(box[4]), float(box[5]), float(box[6])]
                score = float(box[2])
                boxes.append(bbox + [score] + [category])

        if len(boxes) == 0:
            continue
        else:
            boxes = np.array(boxes)

        for class_name, class_id in classname_to_id.items():   # classes that need to merge
            #print(class_id)
            boxes_kind = boxes[boxes[:, -1] == class_id][:, :-1]
            #order_nms = nms(boxes_kind, iou_thr=th_nms)
            order_nms = soft_nms(boxes_kind, iou_thr=th_nms)
            boxes_kind = boxes_kind[order_nms]
            if vote:
                boxes_kind_raw = boxes[boxes[:, -1] == class_id][:, :-1]
                boxes_kind = box_voting(boxes_kind, boxes_kind_raw, th_vote)

            if len(boxes_kind) <= 0:
                continue
            for box in boxes_kind:
                l = {'image_id': img_name,
                     'category_id': class_id,
                     'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                     'score': float(box[4])}
                result_json.append(l)

                result_csv.append(
                    "{},{},{},{},{},{},{}\n".format(class_name, img_name[:-4] + '.xml', box[4], box[0], box[1], box[2], box[3])
                )


    with open(json_output, 'w') as fp:
        json.dump(result_json, fp, indent=4, separators=(',', ': '))

    with open(csv_output, 'w') as file_handler:
        for line in result_csv:
            file_handler.write(line)


if __name__ == '__main__':
    merger_csv()







