from numpy.core.shape_base import hstack
import scipy.io
import cv2 
import os
import matplotlib.pyplot as plt 
import numpy as np 

def shuffle_data(x_data, y_data):
    num_data = len(x_data)
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    shuffled_x_data = [x_data[idx] for idx in indices]    
    shuffled_y_data = [y_data[idx] for idx in indices]    
    return np.array(shuffled_x_data, dtype='float32'), np.array(shuffled_y_data, dtype='float32')

def check_dataset(image_dir, annotation_dir, num_check=-1, shuffle=False):
    image_file_list = os.listdir(image_dir)
    annotation_file_list = os.listdir(annotation_dir)

    # check image and annotation size
    num_image_files = len(image_file_list)
    num_annotation_files = len(annotation_file_list)
    assert num_image_files == num_annotation_files, 'data sizes mismatch'

    if shuffle:
        indices = np.arange(num_image_files)
        np.random.shuffle(indices)
        image_file_list = [image_file_list[idx] for idx in indices]
        annotation_file_list = [annotation_file_list[idx] for idx in indices]

    for ann_file, img_file in zip(annotation_file_list[:num_check], image_file_list[:num_check]):
        img_path = os.path.join(image_dir, img_file)
        ann_path = os.path.join(annotation_dir, ann_file)
        img = cv2.imread(img_path)

        ann = scipy.io.loadmat(ann_path)
        box = ann['box_coord'][0]

        left, top, right, bottom = box[2], box[0], box[3], box[1]

        cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 3)

        plt.imshow(img[:,:,::-1])
        plt.show()

def get_one_hot(num_classes, class_id):
    one_hot = np.zeros(num_classes)
    one_hot[class_id] = 1.0
    return one_hot

def load_class_names(image_dir):
    return os.listdir(image_dir)

def load_dataset(image_dir, annotation_dir, dsize=(224, 224)):

    class_list = os.listdir(image_dir)
    num_classes = len(class_list)

    x_data = []
    y_data = []

    for class_id, class_name in enumerate(class_list):

        image_class_dir = os.path.join(image_dir, class_name)
        annotation_class_dir = os.path.join(annotation_dir, class_name)

        image_file_list = os.listdir(image_class_dir)
        annotation_file_list = os.listdir(annotation_class_dir)

        # check image and annotation size
        num_image_files = len(image_file_list)
        num_annotation_files = len(annotation_file_list)
        assert num_image_files == num_annotation_files, 'data sizes mismatch'

        for ann_file, img_file in zip(annotation_file_list, image_file_list):
            img_path = os.path.join(image_class_dir, img_file)
            ann_path = os.path.join(annotation_class_dir, ann_file)
            
            print('image path      = ', img_path)
            print('annotation path = ', ann_path)

            img = cv2.imread(img_path)            

            height, width = img.shape[:2]
            print('image: height, width = ', height, width)

            ann = scipy.io.loadmat(ann_path)
            box = ann['box_coord'][0]

            left, top, right, bottom = box[2], box[0], box[3], box[1]
            left /= width
            top /= height
            right /= width
            bottom /= height
            ## 
            resized_img = cv2.resize(img, dsize=dsize)
            one_hot = get_one_hot(num_classes, class_id)
            bbox = np.array([left, top, right, bottom])
            print('bbox    = ', bbox)
            print('one hot = ', one_hot)
            gt = np.hstack([bbox, one_hot])


            x_data.append(resized_img.astype('float32') / 255.0)
            y_data.append(gt.astype('float32'))

    return np.array(x_data), np.array(y_data)





if __name__=='__main__':
    IMAGE_DIR = 'D:\\projects\\datasets\\Caltech101\\101_ObjectCategories\\accordion'
    ANNOTATION_DIR = 'D:\\projects\\datasets\\Caltech101\\Annotations\\accordion'
    check_dataset(IMAGE_DIR, ANNOTATION_DIR, num_check=10, shuffle=True)




