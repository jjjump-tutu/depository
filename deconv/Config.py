class Config(object):
    def __init__(self,depth=10,filter_length=32):
        assert (depth - 2) % 4 == 0,"Invalid depth config."
        self.depth = depth
        # self.conv_subsample_lengths = [2, 1] * int((depth - 2)/4)
        self.conv_subsample_lengths = [2, 1, 2 ,1, 2 ,1]
        self.conv_filter_length = filter_length
        self.conv_num_filters_start = 48
        self.conv_init = "he_normal"
        self.conv_activation = "relu"
        self.conv_dropout = 0.5
        self.conv_num_skip = 1
        self.conv_increase_channels_at = 2
        self.batch_size = 256
        self.input_shape = [384, 48]
        self.num_categories = 5
        self.train_epoch = 1
        self.test_epoch = 1

    @staticmethod
    def lr_schedule(epoch):
        lr = 0.005
        if epoch >= 3 and epoch < 5:
            lr = 0.001
        if epoch >= 5:
            lr = 0.0005
        print('Learning rate: ', lr)
        return lr


def construct_config(filter_length = 32):
    config_list = []
    for depth in [14,18,22,34]:
        config_list.append(Config(depth,filter_length))
    return config_list

def construct_config_18(filter_length = 32):
    config_list = []
    for depth in [18]:
        config_list.append(Config(depth,filter_length))
    return config_list