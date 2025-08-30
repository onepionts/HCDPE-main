
import os
import warnings
from argparse import ArgumentParser
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
from model import build_posenet
from tqdm import tqdm
import matplotlib.style as mplstyle
mplstyle.use('fast')
from mmdet.apis import inference_detector, init_detector
has_mmdet = True
class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i] ) /255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i] ) /255.))

color2 = [(252 ,176 ,243) ,(252 ,176 ,243) ,(252 ,176 ,243),
          (0 ,176 ,240), (0 ,176 ,240), (0 ,176 ,240),
          (255 ,255 ,0), (255 ,255 ,0) ,(169, 209, 142),
          (169, 209, 142) ,(169, 209, 142),
          (240 ,2 ,127) ,(240 ,2 ,127) ,(240 ,2 ,127), (240 ,2 ,127), (240 ,2 ,127)]

link_pairs2 = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [9, 7], [7 ,5], [5, 6], [6, 8], [8, 10],
    [3, 1] ,[1, 2] ,[1, 0] ,[0, 2] ,[2 ,4],
]


point_color2 = [(240 ,2 ,127) ,(240 ,2 ,127) ,(240 ,2 ,127),
                (240 ,2 ,127), (240 ,2 ,127),
                (255 ,255 ,0) ,(169, 209, 142),
                (255 ,255 ,0) ,(169, 209, 142),
                (255 ,255 ,0) ,(169, 209, 142),
                (252 ,176 ,243) ,(0 ,176 ,240) ,(252 ,176 ,243),
                (0 ,176 ,240) ,(252 ,176 ,243) ,(0 ,176 ,240),
                (255 ,255 ,0) ,(169, 209, 142),
                (255 ,255 ,0) ,(169, 209, 142),
                (255 ,255 ,0) ,(169, 209, 142)]

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


    # Plot
    fig = plt.figure(figsize=( w /100, h/ 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(image_name[:, :, ::-1])
    bk.set_zorder(-1)

    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        joints_dict = map_joint_dict(dt_joints)

        # stick
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
                ls='-', lw=lw, alpha=1, color=link_pair[2], )
            line.set_zorder(0)
            ax.add_line(line)

        # black ring
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
    # plt.axis('off')
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    # plt.savefig(out_file + '.pdf', format='pdf', bbox_inches='tight', dpi=100)
    plt.savefig(out_file + out_image + '.jpg', format='jpg', bbox_inches='tight', dpi=50)
    plt.close()


def show2Dpose(kps, img):
    if len(kps) == 0:
        return img
    else:
        kps = kps[0]['keypoints']
        # cv2.imwrite(output_dir + 'pose2D/' + str(('%04d' % ii)) + '_2D.png', image)
        # dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        connections1 = [[0, 1], [1, 2], [2, 3], [3, 4]]
        connections2 = [[9, 7], [7, 5], [5, 6], [6, 8], [8, 10]]
        connections3 = [[5, 11], [11, 13], [13, 15]]
        connections4 = [[6, 12], [12, 14], [14, 16]]

        connections5 = [[0, 1], [1, 2], [2, 3], [3, 4], [9, 7], [7, 5], [5, 6], [6, 8], [8, 10], [5, 11], [11, 13],
                        [13, 15],
                        [6, 12], [12, 14], [14, 16]]
        LR1 = np.array([2, 2, 2, 2, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 0])

        connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                       [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                       [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

        LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

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
        # cv2.imwrite()
        # cv2.imwrite(output_dir_2D + str(('%04d' % i)) + '_2D.png', image)
        return img


def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def main():
    det_config = 'F:/HCDPE/tools/vis/cascade_rcnn_x101_64x4d_fpn_coco.py'
    det_checkpoint ='F:/HCDPE/utils/weight/cascade.pth'
    pose_config ='F:/HCDPE/utils/configs/hcdpe_base_classifier.py'
    pose_checkpoint = 'F:/HCDPE/utils/weight/hcdpe.pth'
    show =False
    out_img_root ='F:/HCDPE/demo/output/'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    bbox_thr=0.3
    thickness=1
    det_cat_id=1
    video='F:/HCDPE/demo/01.mp4'


    if out_img_root and not os.path.exists(out_img_root):
        os.makedirs(out_img_root)

    video_path = os.path.join(video)
    # video_path = os.path.join(args.video)
    video_name = video_path.split('/')[-1].split('.')[0]
    # output_dir = './demo/output/' + video_name + '/'
    output_dir = os.path.join(out_img_root) + video_name + '/'
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    print(cap.isOpened())
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_length)

    det_model = init_detector(
        det_config, det_checkpoint, device=device)
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    result = []
    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()

        image_name = frame
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, image_name)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, det_cat_id)

        # test a single image, with a list of bboxes.

        # optional
        return_heatmap = False

        # e.g. use ('backbone', ) to return backbone feature
        output_layer_names = None

        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            image_name,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # 可视化
        if (ii % 1 == 0):
            output_dir_2D = output_dir + 'pose2D_12/'
            os.makedirs(output_dir_2D, exist_ok=True)

            out_file = output_dir_2D
            out_image = str(('%04d' % ii)) + '_2D'

            image = show2Dpose(pose_results, image_name)
            cv2.imwrite(output_dir_2D + str(('%04d' % ii)) + '_2D.jpg', image)

        result.append(pose_results)
    mykeypoint = []

    for i in range(video_length):
        # mykeypoint.append(result[i][0]['keypoints']
        if len(result[i]) == 0:
            temp = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                    [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            mykeypoint.append(temp)
        else:
            mykeypoint.append(result[i][0]['keypoints'][:, 0:2])
    print(mykeypoint)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)
    output_npz = output_dir + 'mykeypoint.npz'
    np.savez_compressed(output_npz, reconstruction=mykeypoint)


if __name__ == '__main__':
    main()
