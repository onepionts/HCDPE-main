
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
from models import build_posenet
from tqdm import tqdm
import matplotlib.style as mplstyle
mplstyle.use('fast')
from mmdet.apis import inference_detector, init_detector
has_mmdet = True
# Define the ColorStyle class and Chunhua_style instance
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

# Function to visualize and save pose results
def vis_pose_result(image_name, pose_results, thickness, out_file):
    h, w, _ = image_name.shape

    # Plot
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = plt.subplot(1, 1, 1)
    bk = plt.imshow(image_name[:, :, ::-1])
    bk.set_zorder(-1)

    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17, -1)
        joints_dict = map_joint_dict(dt_joints)

        # Draw sticks
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11, 16):
                lw = thickness
            else:
                lw = thickness * 2

            line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0], joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1], joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=link_pair[2], )
            line.set_zorder(0)
            ax.add_line(line)

        # Draw black rings
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

    plt.savefig(out_file, format='jpg', bbox_inches='tight', dpi=100)
    plt.close()

# Function to map joints to dictionary
def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)

    return joints_dict

# Function to initialize pose model
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

# Main function to process images
def main():
    input_dir = 'D:/Research/PCT/PCT_standard/data/photo/'  # Update with your input image directory
    output_dir = 'D:/Research/PCT/PCT_standard/data/photos/'  # Update with your desired output directory

    # Check if output directory exists, create if not
    os.makedirs(output_dir, exist_ok=True)

    # Initialize models and other configurations
    det_config = 'D:/Research/PCT/PCT_standard/tools/vis/cascade_rcnn_x101_64x4d_fpn_coco.py'
    det_checkpoint = 'D:/Research/PCT/PCT_standard/utils/weight/cascade.pth'
    pose_config = 'D:/Research/PCT/PCT_standard/utils/configs/hcdpe_base_classifier.py'
    pose_checkpoint = 'D:/Research/PCT/PCT_standard/utils/weight/pct/pct.pth'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize pose model
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
    # Iterate over each image in the input directory
    image_files = sorted(os.listdir(input_dir))
    for filename in tqdm(image_files):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(input_dir, filename)
            image_name = cv2.imread(image_path)

            # Perform pose estimation
            pose_results, _ = inference_top_down_pose_model(pose_model, image_name, format='xywh')

            # Visualize and save pose results
            out_file = os.path.join(output_dir, os.path.splitext(filename)[0] + '_pose.jpg')
            vis_pose_result(image_name, pose_results, thickness=2, out_file=out_file)

if __name__ == '__main__':
    main()
