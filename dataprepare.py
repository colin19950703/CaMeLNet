from glob import glob
from collections import Counter
from torchvision import transforms
import cv2
import torch.utils.data as data
import numpy as np
import pandas as pd
import random

def print_number_of_sample(data_set, prefix):
    def fill_empty_label(label_dict):
        for i in range(max(label_dict.keys()) + 1):
            if label_dict[i] != 0:
                continue
            else:
                label_dict[i] = 0
        return dict(sorted(label_dict.items()))

    data_label = [data_set[i][1] for i in range(len(data_set))]
    d = Counter(data_label)
    d = fill_empty_label(d)
    print("%-7s" % prefix, d)
    data_label = [d[key] for key in d.keys()]

    return data_label

def load_colon(pathname, gt_list=None):
    file_list = glob(pathname)
    label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]

    if gt_list is not None:
        label_list = [gt_list[i] for i in label_list]

    return list(zip(file_list, label_list))

def load_gastric(csv_path, data_dir, data_dir_2, gt_list, nr_claases, down_sample=True):
    def loader(path_list, data_root_dir, gt_list, nr_claases):
        file_list = []
        for tma_name in path_list:
            pathname = glob(f'{data_root_dir}/{tma_name}/*.jpg')
            file_list.extend(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))

        list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < nr_claases]
        return list_out

    df = pd.read_csv(csv_path).iloc[:, :3]
    train_list = list(df.query('Task == "train"')['WSI'])
    valid_list = list(df.query('Task == "val"')['WSI'])
    test_list = list(df.query('Task == "test"')['WSI'])
    train_set = loader(train_list, data_dir, gt_list, nr_claases)

    if down_sample:
        train_normal = [train_set[i] for i in range(len(train_set)) if train_set[i][1] == 0]
        train_tumor = [train_set[i] for i in range(len(train_set)) if train_set[i][1] != 0]

        random.shuffle(train_normal)
        train_normal = train_normal[: len(train_tumor) // 3]
        train_set = train_normal + train_tumor

    valid_set = loader(valid_list, data_dir_2, gt_list, nr_claases)
    test_set = loader(test_list, data_dir_2, gt_list, nr_claases)

    return train_set, valid_set, test_set

def prepare_colon_data(data_root_dir):
    set_1010711 = load_colon('%s/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_colon('%s/1010712/*.jpg' % data_root_dir)
    set_1010713 = load_colon('%s/1010713/*.jpg' % data_root_dir)
    set_1010714 = load_colon('%s/1010714/*.jpg' % data_root_dir)
    set_1010715 = load_colon('%s/1010715/*.jpg' % data_root_dir)
    set_1010716 = load_colon('%s/1010716/*.jpg' % data_root_dir)
    wsi_00016 = load_colon('%s/wsi_00016/*.jpg' % data_root_dir)  # benign exclusively
    wsi_00017 = load_colon('%s/wsi_00017/*.jpg' % data_root_dir)  # benign exclusively
    wsi_00018 = load_colon('%s/wsi_00018/*.jpg' % data_root_dir)  # benign exclusively

    train_set = set_1010711 + set_1010712 + set_1010713 + set_1010715 + wsi_00016
    valid_set = set_1010716 + wsi_00018
    test_set = set_1010714 + wsi_00017

    print_number_of_sample(train_set, 'Train')
    print_number_of_sample(valid_set, 'Valid')
    print_number_of_sample(test_set, 'Test1')
    return train_set, valid_set, test_set

def prepare_colon_test2_data(data_root_dir):
    gt_list = { 0: 5,  # "BN", #0
                1: 0,  # "TLS", #0
                2: 1,  # "TW", #2
                3: 2,  # "TM", #3
                4: 3,  # "TP", #4
                }

    test_set = load_colon('%s/*/*/*.png' % data_root_dir, gt_list)

    print_number_of_sample(test_set, 'Test2')
    return test_set

def prepare_gastric_data(data_root_dir, nr_classes=4):
    """ 8 classes in total only choose 5"""

    if nr_classes == 3:
        gt_train_local = {1: 4,  # "BN", #0
                          2: 4,  # "BN", #0
                          3: 0,  # "TW", #2
                          4: 1,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 4:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 5:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 8,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 6:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 3,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 5,  # "signet", #7
                          10: 5,  # "poorly", #7
                          11: 6  # "LVI", #ignore
                          }
    elif nr_classes == 8:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 3,  # "TM", #3
                          5: 4,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 7,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 10:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 8,  # "poorly", #7
                          11: 9  # "LVI", #ignore
                          }
    else:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 5,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }

    csv_her02 = data_root_dir + '/gastric_wsi/gastric_wsi_PS1024_80_her01_split.csv'
    data_her_dir = data_root_dir + '/gastric_wsi/gastric_wsi_PS1024_80_her01_step05_bright230_resize05'
    data_her_dir_2 = data_root_dir + '/gastric_wsi/gastric_wsi_PS1024_80_her01_step10_bright230_resize05'

    csv_addition = data_root_dir + '/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_split.csv'
    data_add_dir = data_root_dir + '/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step05_bright230_resize05'
    data_add_dir_2 = data_root_dir + '/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step10_bright230_resize05'

    train_set, valid_set, test_set = load_gastric(csv_her02, data_her_dir, data_her_dir_2, gt_train_local, nr_classes)
    train_set_add, valid_set_add, test_set_add = load_gastric(csv_addition, data_add_dir, data_add_dir_2, gt_train_local, nr_classes, down_sample=False)
    train_set += train_set_add
    valid_set += valid_set_add
    test_set += test_set_add

    print_number_of_sample(train_set, 'Train')
    print_number_of_sample(valid_set, 'Valid')
    print_number_of_sample(test_set, 'Test')

    return train_set, valid_set, test_set

class DatasetSerial(data.Dataset):
    def __init__(self, pair_list, shape_augs=None, input_augs=None):
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):
        pair = self.pair_list[idx]
        # print(pair)
        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1]
        # print(input_img.shape)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.],
                                 std=[1., 1., 1.])
        ])

        if self.shape_augs is not None:
            shape_augs = self.shape_augs.to_deterministic()
            input_img = shape_augs.augment_image(input_img)

        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        input_img = np.array(input_img).copy()
        out_img = np.array(transform(input_img)).transpose(1, 2, 0)

        return np.array(out_img), img_label

    def __len__(self):
        return len(self.pair_list)

if __name__ == '__main__':
    print('\nColoectal')
    prepare_colon_data()
    prepare_colon_test2_data()

    print('\nGastric')
    prepare_gastric_data()