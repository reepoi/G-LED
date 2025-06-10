import argparse
import logging
import os
import pprint
import sys
from datetime import datetime

import hydra
import torch
from omegaconf import OmegaConf

from conf import conf
from g_led import utils
from g_led.data import data_bfs_preprocess
from g_led.train_test_seq.train_seq import train_seq_shift
from g_led.transformer.sequentialModel import SequentialModel as transformer

log = logging.getLogger(__file__)


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('hydra', nargs='*')
        """
        for dataset
        """
        self.parser.add_argument("--dataset",
                                 default='bfs_les',
                                 help='name it')
        self.parser.add_argument("--data_dir", default='/mnta/taosData/diffusion-dynamics/G-LED/data')
        self.parser.add_argument("--data_location",
                                 default = ['/root/workspace/out/diffusion-dynamics/G-LED/data/data0.npy',
                                            '/root/workspace/out/diffusion-dynamics/G-LED/data/data1.npy'],
                                 help='the relative or abosolute data.npy file')
        self.parser.add_argument("--trajec_max_len",
                                 default=41,
                                 help = 'max seq_length (per seq) to train the model')
        self.parser.add_argument("--start_n",
                                 default=0,
                                 help = 'the starting step of the data')
        self.parser.add_argument("--n_span",
                                 default=8000,
                                 help='the total step of the data from the staring step')



        self.parser.add_argument("--trajec_max_len_valid",
                                 default=450,
                                 help = 'max seq_length (per seq) to valid the model')
        self.parser.add_argument("--start_n_valid",
                                 default=8000,
                                 help = 'the starting step of the data')
        self.parser.add_argument("--n_span_valid",
                                 default=500,
                                 help='the total step of the data from the staring step')


        """
        for model
        """
        self.parser.add_argument("--n_layer",
                                 default =8,#8
                                 help = 'number of attention layer')
        self.parser.add_argument("--output_hidden_states",
                                 default= True,
                                 help='out put hidden matrix')
        self.parser.add_argument("--output_attentions",
                                 default = True,
                                 help = 'out put attention matrix')
        self.parser.add_argument("--n_ctx",
                                 default = 40,
                                 help='number steps transformer can look back at')
        self.parser.add_argument("--n_embd",
                                 default = 2048,
                                 help='The hidden state dim transformer to predict')
        self.parser.add_argument("--n_head",
                                 default = 4,
                                 help='number of head per layer')
        self.parser.add_argument("--embd_pdrop",
                                 default = 0.0,
                                 help='T.B.D')
        self.parser.add_argument("--layer_norm_epsilon",
                                 default=1e-5,
                                 help='############ Do not change')
        self.parser.add_argument("--attn_pdrop",
                                 default = 0.0,
                                 help='T.B.D')
        self.parser.add_argument("--resid_pdrop",
                                 default = 0.0,
                                 help='T.B.D')
        self.parser.add_argument("--activation_function",
                                 default = "relu",
                                 help='Trust OpenAI and Nick')
        self.parser.add_argument("--initializer_range",
                                 default = 0.02,
                                 help='Trust OpenAI and Nick')


        """
        for training
        """
        self.parser.add_argument("--start_Nt",
                                 default=1,
                                 help='The starting length of forward propgatate')
        self.parser.add_argument("--d_Nt",
                                 default=1,
                                 help='The change length of forward propgatate')
        self.parser.add_argument("--batch_size",
                                 default=16, #max 16->0.047
                                 help = 'how many seqs you want to train together per bp')
        self.parser.add_argument("--batch_size_valid",
                                 default=16, #max 16->0.047
                                 help = 'how many seqs you want to train together per valid')
        self.parser.add_argument("--shuffle",
                                 default=True,
                                 help = 'shuffle the batch')
        self.parser.add_argument("--device",
                                 default='cuda:0')
        self.parser.add_argument("--epoch_num",
                                 default = 10000,
                                 help='epoch_num')
        self.parser.add_argument("--learning_rate",
                                 default = 1e-4,
                                 help='learning rate')
        self.parser.add_argument("--gamma",
                                 default=0.99083194489,
                                 help='learning rate decay')

        self.parser.add_argument("--coarse_dim",
                                 default=[32,32],
                                 help='the coarse shape (hidden) of transformer')
        self.parser.add_argument('--coarse_mode',
                                 default='bilinear',
                                 help='the way of downsampling the snpashot')
        self.parser.add_argument("--march_tol",
                                 default=0.01,
                                 help='march threshold for Nt + 1')

    def update_args(self):
        args = self.parser.parse_args()
        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        # output dataset
        args.dir_output = '/home/sci/ttransue/out/g_led/runs/original_code'
        args.fname = args.dataset + '_' +args.time
        args.experiment_path = args.dir_output + args.fname
        args.model_save_path = args.experiment_path + '/' + 'model_save/'
        args.logging_path = args.experiment_path + '/' + 'logging/'
        args.current_model_save_path = args.model_save_path
        args.logging_epoch_path = args.logging_path + 'epoch_history.csv'
        if not os.path.isdir(args.logging_path):
            os.makedirs(args.logging_path)
        if not os.path.isdir(args.model_save_path):
            os.makedirs(args.model_save_path)
        return args








@hydra.main(**utils.HYDRA_INIT)
def main(cfg):
    engine = conf.get_engine()
    conf.orm.create_all(engine)
    with conf.sa.orm.Session(engine) as db:
        cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        log.info('Command: python %s', ' '.join(sys.argv))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    # assert args.coarse_dim[0]*args.coarse_dim[1]*2 == args.n_embd
    # assert args.trajec_max_len_valid == args.n_ctx + 1

    dataset = data_bfs_preprocess.get_dataset(cfg.dataset)
    dataset.setup('fit')
    data_loader_train = dataset.train_dataloader()
    dl_val = dataset.val_dataloader(combined=False)
    data_loader_test_on_train = dl_val['val_on_train']
    data_loader_valid = dl_val['val']
    """
    create model
    """
    model = transformer(cfg.model, 2 * cfg.dataset.embedding_dimension).float().to(cfg.device)
    print('Number of parameters: {}'.format(model._num_parameters()))

    """
    create loss function
    """
    loss_func = torch.nn.MSELoss()

    """
    create optimizer
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    """
    create scheduler
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.model.learning_rate_decay)
    """
    train
    """
    train_seq_shift(cfg=cfg,
                    model=model,
                    data_loader=data_loader_train,
                    data_loader_copy=data_loader_test_on_train,
                    data_loader_valid=data_loader_valid,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    scheduler=scheduler)


if __name__ == '__main__':
    last_override, run_dir = utils.get_run_dir()
    utils.set_run_dir(last_override, run_dir)
    main()
