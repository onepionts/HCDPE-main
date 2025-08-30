import os
import warnings
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results, collect_multi_frames)
from mmpose.datasets import DatasetInfo

from models import build_posenet
import matplotlib.style as mplstyle
mplstyle.use('fast')
from mmdet.apis import inference_detector, init_detector

class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))

color2 = [(252, 176, 243), (252, 176, 243), (252, 176, 243),
          (0, 176, 240), (0, 176, 240), (0, 176, 240),
          (255, 255, 0), (255, 255, 0), (169, 209, 142),
          (169, 209, 142), (169, 209, 142),
          (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127), (240, 2, 127)]

link_pairs2 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [9, 7], [7, 5], [5, 6], [6, 8], [8, 10],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
]

point_color2 = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
                (240, 2, 127), (240, 2, 127),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (252, 176, 243), (0, 176, 240), (252, 176, 243),
                (0, 176, 240), (252, 176, 243), (0, 176, 240),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142),
                (255, 255, 0), (169, 209, 142)]

chunhua_style = ColorStyle(color2, link_pairs2, point_color2)

def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
    return joints_dict

def vis_pose_result(image_name, out_image, pose_results, thickness, out_file):
    h = image_name.shape[0]
    w = image_name.shape[1]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(image_name[:, :, ::-1])
    bk.set_zorder(-1)

    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        joints_dict = map_joint_dict(dt_joints)
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11, 16):
                lw = thickness
            else:
                lw = thickness * 2
            line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                          joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                          joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2])
            line.set_zorder(0)
            ax.add_line(line)
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2
            circle = mpatches.Circle(tuple(dt_joints[k, :2]),
                                     radius=radius,
                                     ec='black',
                                     fc=chunhua_style.ring_color[k],
                                     alpha=1,
                                     linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(out_file + out_image + '.jpg', format='jpg', bbox_inches='tight', dpi=50)
    plt.close()

def show2Dpose(kps, img):#画图用的
    if len(kps) == 0:
        return img
    else:
        kps = kps[0]['keypoints']
        connections5 = [[0, 1], [1, 2], [2, 3], [3, 4], [9, 7], [7, 5], [5, 6], [6, 8], [8, 10], [5, 11], [11, 13],
                        [13, 15],
                        [6, 12], [12, 14], [14, 16]]
        LR1 = np.array([2, 2, 2, 2, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 0])
        lcolor = (255, 0, 0)
        rcolor = (0, 0, 255)
        gcolor = (25, 200, 200)
        thickness = 3
        for j, c in enumerate(connections5):
            start = map(int, kps[c[0]])
            end = map(int, kps[c[1]])
            start = list(start)
            end = list(end)
            if LR1[j] == 2:
                color = gcolor
            elif LR1[j] == 1:
                color = rcolor
            else:
                color = lcolor
            cv2.line(img, (start[0], start[1]), (end[0], end[1]), color, thickness=10)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=8)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=8)
        return img

def init_pose_model(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model

def main():
    det_config = '/home/HCDPE/tools/vis/cascade_rcnn_x101_64x4d_fpn_coco.py'
    det_checkpoint = '/home/HCDPE/utils/weight/cascade.pth'
    pose_config = '/home/HCDPE/utils/configs/hcdpe_base_classifier.py'
    pose_checkpoint = '/home/HCDPE/weights/classifier_coco_12.17_210(40,2048,mh).pth'
    image_path = r'/home/HCDPE/data/coco/images/val2017/val2017/000000003156.jpg'  # Path to the input image
    out_img_root = '/home/HCDPE/demo/output/aaa'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    bbox_thr = 0.3
    thickness = 1
    det_cat_id = 1

    if out_img_root and not os.path.exists(out_img_root):
        os.makedirs(out_img_root)

    det_model = init_detector(det_config, det_checkpoint, device=device)
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    dataset_info = DatasetInfo(dataset_info)

    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    
    mmdet_results = inference_detector(det_model, img)
    
    person_results = process_mmdet_results(mmdet_results, det_cat_id)
   
    pose_results,_ = inference_top_down_pose_model(#result['preds']& result['output_heatmap']
        pose_model,
        img,
        person_results,
        bbox_thr=bbox_thr,
        format='xyxy',
        dataset=pose_model.cfg.data.test.type,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None
    )


    img = show2Dpose(pose_results, img)
    vis_pose_result(img, os.path.splitext(image_name)[0], pose_results, thickness, out_img_root)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
