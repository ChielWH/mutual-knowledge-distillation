import os
import sys
import cv2
import glob
import shutil
import fnmatch
import subprocess
from tqdm import tqdm

# heavily based on https://github.com/tjmoon0104/pytorch-tiny-imagenet

os.chdir('data')

# download and unzip the dataset
subprocess.call('wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip;\
                unzip tiny-imagenet-200.zip;\
                rm -r ./tiny-imagenet-200/test',
                shell=True)

target_folder = './tiny-imagenet-200/val/'
test_folder = './tiny-imagenet-200/test/'

# storing the filename - label pairs
os.mkdir(test_folder)
val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

# structure the folders in the right format for PyTorch according their labels
paths = glob.glob('./tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')
    if not os.path.exists(test_folder + str(folder)):
        os.mkdir(test_folder + str(folder))
        os.mkdir(test_folder + str(folder) + '/images')

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if len(glob.glob(target_folder + str(folder) + '/images/*')) < 25:
        dest = target_folder + str(folder) + '/images/' + str(file)
    else:
        dest = test_folder + str(folder) + '/images/' + str(file)
    shutil.move(path, dest)

os.rmdir('./tiny-imagenet-200/val/images')

# removing the .txt files containing the bounding boxes
for root, dirnames, filenames in os.walk('tiny-imagenet-200'):
    for filename in fnmatch.filter(filenames, '*.txt'):
        os.remove(os.path.join(root, filename))

# resizing the images from 200*200 to 224*224
shutil.copytree('tiny-imagenet-200', 'tiny-imagenet-224')
all_images = glob.glob('tiny-imagenet-224/*/*/*/*')

if sys.stdout.isatty():  # if in terminal
    with tqdm(total=len(all_images)) as pbar:
        pbar.set_description(
            f'resizing the images from 200 by 200 to 224 by 224')
        for image_path in all_images:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image_path, img)
            pbar.update(1)

else:
    print('resizing the images from 200 by 200 to 224 by 224')
    for idx, image_path in enumerate(all_images, 1):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_path, img)
        if idx % (len(all_images) // 10) == 0:
            print(f'at {int(idx / len(all_images) * 100)}%')
    print('resizing done')

for image_path in all_images:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_path, img)

os.rmdir('tiny-imagenet-200.zip')
