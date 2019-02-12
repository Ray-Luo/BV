from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/home/kdd/Documents/simple-faster-rcnn-pytorch/BV_devkit/all_data'
    # min_size = 3956  # image resize
    # max_size = 5280 # image resize
    min_size = 1000  # image resize 3k x 4k --> 0.134
    max_size = 1000 # image resize 2k x 3k --> 0.210
    # image resize 1k x 1.3k --> 0.436
    num_workers = 0
    test_num_workers = 0

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-4


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 1  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 70


    use_adam = True # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = True # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
