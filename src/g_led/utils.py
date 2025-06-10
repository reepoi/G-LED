import argparse
import json
import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt

from conf import conf


DIR_ROOT = (Path(__file__).parent/'..'/'..').resolve()
HYDRA_INIT = dict(version_base=None, config_path='../../conf', config_name='conf')


def get_run_dir(hydra_init=HYDRA_INIT, commit=True, engine_name='runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = conf.get_engine(name=engine_name)
        conf.orm.create_all(engine)
        with conf.sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


def get_data_location(args):
    if args.dataset == 'ins_channel':
        data_location = os.path.join(args.data_location, 'data_set_ins')
    elif args.dataset == 'backward_facing':
        data_location = os.path.join(args.data_location, 'data_set_pitz')
    elif args.dataset == 'duan':
        data_location = os.path.join(args.data_location, 'data_set_duan')
    else:
        raise ValueError('Not implemented')
    return data_location


def save_loss(args, loss_list, Nt):
    plt.figure()
    plt.plot(loss_list,'-o')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(str(min(loss_list))+'Nt'+str(Nt))
    print(os.path.join(args.logging_path, 'loss_curve.png'))
    plt.savefig(os.path.join(args.logging_path, 'loss_curve.png'))
    plt.close()
    np.savetxt(os.path.join(args.logging_path, 'loss_curve.txt'),
                np.asarray(loss_list))

def save_args(args):
    with open(os.path.join(args.logging_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def save_args_sample(args,name):
    with open(os.path.join(args.experiment_path, name), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def read_args_txt(args, argtxt):
    #args.parser.parse_args(namespace=args.update_args_no_folder_create())
    f = open (argtxt, "r")
    args = args.parser.parse_args(namespace=argparse.Namespace(**json.loads(f.read())))
    return args
    return t

def save_model(model, model_save_path, Nt, bestModel = False):
    if bestModel:
        torch.save(model.state_dict(), model_save_path/'best_model_sofar')
        np.savetxt(model_save_path/'best_model_sofar_Nt', np.ones(2) * Nt)
    else:
        torch.save(model.state_dict(), model_save_path/f'model_epoch_{Nt}')

def load_model(model,args_train,args_sample):
    if args_sample.usebestmodel:
        model.load_state_dict(torch.load(args_train.current_model_save_path+'best_model_sofar'))
    else:
        model.load_state_dict(torch.load(args_train.current_model_save_path+'model_epoch_'+str(args_sample.model_epoch)))
    return model














class normalizer_1dks(object):
    """
    arguments:
    target_dataset (torch.utils.data.Dataset) : this is dataset we
                                                want to normalize
    """
    def __init__(self, target_dataset,args):
        # mark the orginal device of the target_dataset
        self.mean = target_dataset.mean().to(args.device)
        self.std  = target_dataset.std().to(args.device)
    def normalize(self, batch):
        return (batch - self.mean) / self.std
    def normalize_inv(self, batch):
        return batch * self.std +self.mean




















if __name__ == '__main__':
    num_videos = 10
    fig, axs = plt.subplots(2,int(num_videos/2))
    number_of_sample = int(num_videos/2)
    fig.subplots_adjust(hspace=-0.9,wspace=0.1)
    videos_to_plot = [np.zeros([1,3,1,64,256]) for _ in range(num_videos)]
    j = 0
    for k in range(0, num_videos):
        this_video = videos_to_plot[k-1]
        axs[k//number_of_sample, k%number_of_sample].imshow(np.sqrt(this_video[0,0,j,:,:]**2 + this_video[0,1,j,:,:]**2))
        axs[k//number_of_sample, k%number_of_sample].set_xticks([])
        axs[k//number_of_sample, k%number_of_sample].set_yticks([])
    plt.savefig('test_space.png',bbox_inches='tight')
