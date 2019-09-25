"""

"""


# Built-in
import os
import json
from glob import glob

# Libs
import albumentations as A
import matplotlib.pyplot as plt
from natsort import natsorted
from albumentations.pytorch import ToTensor

# Own modules
from data import data_utils
from utils import misc_utils
from network import StackMTLNet, network_utils

# Settings
MODEL_DIR = r'/hdd6/Models/line_mtl_exp/20190924_154508'
EPOCH_NUM = 99
GPU = '1'


def main():
    config = json.load(open(os.path.join(MODEL_DIR, 'config.json')))

    # set gpu
    device, parallel = misc_utils.set_gpu(GPU)
    model = StackMTLNet.StackHourglassNetMTL(config['task1_classes'], config['task2_classes'], config['backbone'])
    network_utils.load(model, os.path.join(MODEL_DIR, 'epoch-{}.pth.tar'.format(EPOCH_NUM)), disable_parallel=True)
    model.to(device)
    model.eval()

    # eval on dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensor(sigmoid=False),
    ])
    save_dir = os.path.join(r'/hdd/Results/line_mtl_cust', os.path.basename(MODEL_DIR))
    evaluator = network_utils.Evaluator('transmission', tsfm_valid, device)
    evaluator.evaluate(model, (512, 512), 0,
                       pred_dir=save_dir, report_dir=save_dir, save_conf=True)


def compare_results():
    results_dir = os.path.join(r'/hdd/Results/line_mtl_cust', os.path.basename(MODEL_DIR))
    image_dir = r'~/Documents/bohao/data/transmission_line/raw2'
    pred_files = natsorted(glob(os.path.join(results_dir, '*.png')))
    conf_files = natsorted(glob(os.path.join(results_dir, '*.npy')))
    for pred_file, conf_file in zip(pred_files, conf_files):
        file_name = os.path.splitext(os.path.basename(pred_file))[0]
        rgb_file = os.path.join(image_dir, '{}.tif'.format(file_name))
        rgb = misc_utils.load_file(rgb_file)[..., :3]
        pred = misc_utils.load_file(pred_file)
        # make line map
        lbl_file = os.path.join(r'/media/ei-edl01/data/remote_sensing_data/transmission_line/parsed_annotation',
                                '{}.csv'.format(file_name))
        boxes, lines = data_utils.csv_to_annotation(lbl_file)
        lbl, _ = data_utils.render_line_graph(rgb.shape[:2], boxes, lines)

        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot(131)
        plt.imshow(rgb)
        plt.axis('off')
        plt.subplot(132, sharex=ax1, sharey=ax1)
        plt.title(file_name.replace('_resize', ''))
        plt.imshow(lbl)
        plt.axis('off')
        plt.subplot(133, sharex=ax1, sharey=ax1)
        plt.imshow(pred)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
    # compare_results()
