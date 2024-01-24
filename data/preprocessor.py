from PIL import Image
import os
import argparse
from sklearn.model_selection import train_test_split
import pydicom
import numpy as np
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--in', required=True, help='input folder')
#ap.add_argument('-o', '--out', required=True, help='output folder')
ap.add_argument('-t', '--test', required=True, help='proportion of images used for test')
ap.add_argument('-v', '--validation', required=True, help='proportion of images used for validation')
ap.add_argument('-d', '--dimension', required=True, help='new dimension for files')
args = vars(ap.parse_args())


def preprocess(in_path, out_path, test_size, val_size, dim):
    dirs = ['0','1']
    for label in tqdm(dirs):
        label_path = os.path.join(in_path, label)
        print("label_path:",label_path)
        assert (os.path.isdir(label_path))

        train_path = os.path.join(out_path, 'train', label)
        os.makedirs(train_path, exist_ok=True)
        test_path = os.path.join(out_path, 'test', label)
        os.makedirs(test_path, exist_ok=True)
        val_path = os.path.join(out_path, 'validation', label)
        os.makedirs(val_path, exist_ok=True)

        images = os.listdir(label_path)

        train, test_val = train_test_split(images, test_size=test_size + val_size)
        test, val = train_test_split(test_val, test_size=val_size / (test_size + val_size))

        resize(label_path, train_path, train, dim)
        resize(label_path, test_path, test, dim)
        resize(label_path, val_path, val, dim)


def resize(in_path, out_path, images, dim):
    for image in tqdm(images):
        image_in_path = os.path.join(in_path, image)
        assert (os.path.isfile(image_in_path))
        image_out_path = os.path.join(out_path, f"{os.path.splitext(image)[0]}.png")

        # im = Image.open(image_in_path)
        # im_resized = im.resize((dim, dim), Image.ANTIALIAS)
        # im_resized.save(image_out_path, 'png', quality=100)

        #read DICOM image and save to png
        ds = pydicom.dcmread(image_in_path)
        new_image = ds.pixel_array.astype(float)
        scaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0

        scaled_image = np.uint8(scaled_image)
        final_image = Image.fromarray(scaled_image)
        im_resized = final_image.resize((dim, dim), Image.LANCZOS)  #ANTIALIAS changed to LANCZOS in new version
        im_resized.save(image_out_path, 'png', quality=100)

if __name__ == "__main__":
    preprocess(args['in'], os.path.join('.'), int(args['test']), int(args['validation']), int(args['dimension']))
