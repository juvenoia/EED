import os
import random

random.seed(42)

# training

url = "/home/wbo/CVPR25/MSEC/validation/"

suffix = [
    "N1.5",
    "N1",
    "P1.5",
    "P1"
]

gt = url + "GT_IMAGES/"
lq = url + "INPUT_IMAGES/"

for img in os.listdir(gt):
    gt_url = img
    for var in suffix:
        lq_url = img[:-4] + "_" + var + ".JPG"
        print(lq + lq_url + ' ' + gt + gt_url)
