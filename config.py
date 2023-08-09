import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa
####

class CONFIGURE(object):
    def __init__(self, _args=None):
        # if _args is not None:
        #     self.__dict__.update(_args.__dict__)
        self.seed = 5
        self.init_lr = 1.0e-3
        self.lr_steps = 20  # decrease at every n-th epoch
        self.gamma = 0.2
        self.nr_epochs = 50
        self.nr_classes = 4

        # nr of processes for parallel processing input
        self.nr_procs_train = 8
        self.nr_procs_valid = 8

        self.input_size = 512
        self.train_batch_size = 64
        self.infer_batch_size = 64
        self.feat_dim = 1280

        self.load_network = False
        self.save_net_path = ""
        self.logging = True  # True for debug run only

        self.log_path = 'log/'
        self.chkpts_prefix = 'model'

        if _args is not None:
            self.__dict__.update(_args.__dict__)
            if self.data_name == 'colon':
                self.data_dir = 'data/colon/KBSMC_colon_tma_cancer_grading_1024/'
                self.data_dir2 = 'data/colon/KBSMC_colon_45wsis_cancer_grading_512 (Test 2)/'
                self.pretrained_dir = 'pretrained/CaMeLNet_colon.pth'
                self.epoch_length = None
            elif self.data_name == 'gastric':
                self.data_dir = 'data/gastric/'
                self.data_dir2 = ''
                self.pretrained_dir = 'pretrained/CaMeLNet_gastric.pth'
                self.epoch_length = 100

        self.log_dir = self.log_path


    def train_augmentors(self):
        shape_augs = [
            iaa.Resize({"height": self.input_size, "width": self.input_size}, interpolation='nearest')
        ]
        #
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        input_augs = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            iaa.Resize({"height": self.input_size, "width": self.input_size}, interpolation='nearest')
        ]

        return shape_augs, None
