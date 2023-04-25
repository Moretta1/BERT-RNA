import sys
import os
import pickle

# this is a demo for the training procedure
# you may modify it according to your need

from configuration import config_init
from frame import Learner


def SL_train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    roc_datas, prc_datas = [], []

    # 
    if config.model == 'FusionDNAbert':
        config.kmers = [4, 6]

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    # learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()



def SL_finetune():
    # remember to change it to your path
    config = pickle.load(open('../result/trainCross/3mer/config.pkl', 'rb'))
    config.path_params = '../result/trainCross/3mer/config.txt'
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()

def select_dataset():
    train_dict = {
        # training sets
        "m1AS": '../data/RNA_MS/tsv/m1A/m1A_sacCer3/train.tsv',
        "m1AM": '../data/RNA_MS/tsv/m1A/m1A_mm10/train.tsv',
        "m1AH": '../data/RNA_MS/tsv/m1A/m1A_hg19/train.tsv',
        "m6AS": '../data/RNA_MS/tsv/m6A/m6A_sacCer3/train.tsv',
        "m6AB": '../data/RNA_MS/tsv/m6A/m6A_BDGP6/train.tsv',
        "m6AD": '../data/RNA_MS/tsv/m6A/m6A_danRer10/train.tsv',
        "m6AE": '../data/RNA_MS/tsv/m6A/m6A_E.coli/train.tsv',
        "m6AH": '../data/RNA_MS/tsv/m6A/m6A_hg19/train.tsv',
        "m6AP": '../data/RNA_MS/tsv/m6A/m6A_P.aeruginosa/train.tsv',
        "m6APAN": '../data/RNA_MS/tsv/m6A/m6A_panTro4/train.tsv',
        "m6AR": '../data/RNA_MS/tsv/m6A/m6A_rheMac8/train.tsv',
        "m6ARN5": '../data/RNA_MS/tsv/m6A/m6A_rn5/train.tsv',
        "m6ASUR": '../data/RNA_MS/tsv/m6A/m6A_susScr3/train.tsv',
        "m6AT": '../data/RNA_MS/tsv/m6A/m6A_TAIR10/train.tsv',
        "pseS": '../data/RNA_MS/tsv/pseudoU/pseudoU_sacCer3/train.tsv',
        "pseM": '../data/RNA_MS/tsv/pseudoU/pseudoU_mm10/train.tsv',
        "pseH": '../data/RNA_MS/tsv/pseudoU/pseudoU_hg19/train.tsv',
    }

    test_dict = {
     # testing sets
        "m1AS": '../data/RNA_MS/tsv/m1A/m1A_sacCer3/test.tsv',
        "m1AM": '../data/RNA_MS/tsv/m1A/m1A_mm10/test.tsv',
        "m1AH": '../data/RNA_MS/tsv/m1A/m1A_hg19/test.tsv',
        "m6AS": '../data/RNA_MS/tsv/m6A/m6A_sacCer3/test.tsv',
        "m6AB": '../data/RNA_MS/tsv/m6A/m6A_BDGP6/test.tsv',
        "m6AD": '../data/RNA_MS/tsv/m6A/m6A_danRer10/test.tsv',
        "m6AE": '../data/RNA_MS/tsv/m6A/m6A_E.coli/test.tsv',
        "m6AH": '../data/RNA_MS/tsv/m6A/m6A_hg19/test.tsv',
        "m6AP": '../data/RNA_MS/tsv/m6A/m6A_P.aeruginosa/test.tsv',
        "m6APAN": '../data/RNA_MS/tsv/m6A/m6A_panTro4/test.tsv',
        "m6AR": '../data/RNA_MS/tsv/m6A/m6A_rheMac8/test.tsv',
        "m6ARN5": '../data/RNA_MS/tsv/m6A/m6A_rn5/test.tsv',
        "m6ASUR": '../data/RNA_MS/tsv/m6A/m6A_susScr3/test.tsv',
        "m6AT": '../data/RNA_MS/tsv/m6A/m6A_TAIR10/test.tsv',
        "pseS": '../data/RNA_MS/tsv/pseudoU/pseudoU_sacCer3/test.tsv',
        "pseM": '../data/RNA_MS/tsv/pseudoU/pseudoU_mm10/test.tsv',
        "pseH": '../data/RNA_MS/tsv/pseudoU/pseudoU_hg19/test.tsv',       
    }
    # print(sys.argv)
    # the input positive for training and testing files are fixed
    path_train_data = train_dict[sys.argv[2]] 
    path_test_data = test_dict[sys.argv[4]]

    return path_train_data, path_test_data


if __name__ == '__main__':
    config = config_init.get_config()
    print("train:" + config.path_train_data)
    print("test:" + config.path_test_data)
    config.path_train_data, config.path_test_data = select_dataset()
    SL_train(config)

