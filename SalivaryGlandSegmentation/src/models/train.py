#----------
# Libraries for importing files
#----------
import sys
import os
import yaml
from yaml.loader import SafeLoader

#----------
# Local run
#----------

# sys.path.append('C:\\Users\\Natalia\\Documents\\01. Master\\2022-SS\\01. PMSD\\07. Code\\SalivaryGlandSegmentation\\src\\utils')
# sys.path.append('C:\\Users\\Natalia\\Documents\\01. Master\\2022-SS\\01. PMSD\\07. Code\\SalivaryGlandSegmentation\\src\\data')
# sys.path.append('C:\\Users\\Natalia\\Documents\\01. Master\\2022-SS\\01. PMSD\\07. Code\\SalivaryGlandSegmentation\\src\\metrics')
# CONFIG_PATH = 'C:\\Users\\Natalia\\Documents\\01. Master\\2022-SS\\01. PMSD\\07. Code\\SalivaryGlandSegmentation\\config'


#-------------
# Polyaxon run 
#-------------
sys.path.append('./src/utils')
sys.path.append('./src/data')
sys.path.append('./src/metrics')
CONFIG_PATH = './config'

from polyaxon_client.tracking import Experiment

if __name__ == "__main__":


    with open(os.path.join(CONFIG_PATH, 'config_dncnn.yaml')) as f:
        hparams = yaml.load(f, Loader=SafeLoader)
    
    if hparams['on_polyaxon']:
        
        experiment = Experiment()
        data_paths = experiment.get_data_paths()
        root_files = data_paths['data1']+ hparams['polyaxon_dataset']

        gpus = hparams["gpus_polyaxon"]
        num_workers = hparams["num_workers_polyaxon"]
    else:
        root_files = hparams['local_dataset']
        gpus = hparams["gpus_local"]
        num_workers = hparams["num_workers_local"]

    print(root_files)
    #     # Define some variables from yaml file
    #     if experiment.get_declarations()["data_root"] is None:
    #         data_paths = experiment.get_data_paths()
    #         print(f' >>> Path of data in the NAS is under : {data_paths}')
    #         hparams['data_root'] = data_paths['data1']
    #     else:
    #         hparams['data_root'] = experiment.get_declarations()["data_root"]
    #         hparams['data_root'] = 'data'

    #     hparams['path_to_data'] = os.path.join(hparams['data_root'], hparams['dataset'])

    #     print("Data is located in " + hparams['path_to_data'])

    # else:


    #     print("Data is located in " + hparams['path_to_data'])


    # print('Output path: ', hparams['out_dir'])

#----------
# Libraries for training
#----------

from data_generator import SalivaryGlandsDataset
from metrics_evaluation import *
from unet import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from segmentation import Segmentation
from metrics_evaluation import *
import pytorch_lightning as pl
import torch
import numpy as np

torch.cuda.empty_cache()

def train(hparams, path, face_side, network, gpus, num_workers):

    ####################
    ### Load Data
    ####################
    if network == 'net1':
        net2 = False
        cut_head = hparams['net1_cut_head'] 
        lr=hparams['net1_learning_rate']
        monitor=hparams["net1_monitoring_metric"] 
        min_delta=hparams["net1_min_delta"] 
        patience=hparams["net1_patience"] 
        verbose=hparams["net1_verbose"]
        mode=hparams["net1_mode"]
        check_val_every_n_epoch = hparams["net1_check_val_every_epoch"]
        log_every_n_steps = hparams["net1_log_every_step"]
        model = unet3_net1
        batch_size = hparams["batch_size_bb"]
    else:
        net2 = True
        cut_head = hparams['net2_cut_head'] 
        lr=hparams['net2_learning_rate']
        monitor=hparams["net2_monitoring_metric"] 
        min_delta=hparams["net2_min_delta"] 
        patience=hparams["net2_patience"] 
        verbose=hparams["net2_verbose"]
        mode=hparams["net2_mode"]
        check_val_every_n_epoch = hparams["net2_check_val_every_epoch"]
        log_every_n_steps = hparams["net2_log_every_step"]
        model = unet3_net2
        batch_size = hparams["batch_size_seg"]

    
    train_dataset = SalivaryGlandsDataset(root_dir = path, 
                                        split = 'train', 
                                        transform = None, 
                                        cut_head = cut_head, 
                                        net2 = net2, 
                                        side = face_side, 
                                        resample = hparams['resample_data'])  
    
    val_dataset = SalivaryGlandsDataset(root_dir = path, 
                                        split = 'val',
                                        transform = None, 
                                        cut_head = cut_head, 
                                        net2 = net2, 
                                        side = face_side,
                                        resample = hparams['resample_data']) 


    ####################
    ### Load Dataloaders
    ####################

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    writer = SummaryWriter('runs')

    ####################
    ### Define the model of network
    ####################

    net = Segmentation(model = model,  
                        lr=lr, 
                        side = face_side, 
                        net2 = net2)


    # Define how and when to save your models during training
    checkpoint_callback = [EarlyStopping(monitor=monitor, 
                                        min_delta=min_delta, 
                                        patience=patience, 
                                        verbose=verbose,
                                        mode=mode)]



    trainer = pl.Trainer(gpus=gpus, 
                    callbacks=[EarlyStopping(monitor="val_dice", mode="max")],
                    check_val_every_n_epoch = check_val_every_n_epoch,
                    log_every_n_steps = log_every_n_steps,
                    )

    # Train!
    trainer.fit(net, train_loader, val_loader)

    # Save model
    #torch.save(net.state_dict(), 'models/model_'+network+'_' + face_side + '.pt')

    # 
    del train_dataset, train_loader, val_dataset, val_loader

    test_dataset = SalivaryGlandsDataset(root_dir = path, 
                                        split = 'test', 
                                        transform = None, 
                                        cut_head = cut_head, 
                                        net2 = net2, 
                                        side = face_side,
                                        resample = hparams['resample_data'])  

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    trainer.test(net, dataloaders=test_loader)



# # ---------------------
# RUN TRAINING
# ---------------------
net1_right = train(hparams, path = root_files, face_side='right',network='net1', gpus=1, num_workers = num_workers)
net1_right

# net1_left = train(hparams,path = root_files, face_side='left',network='net1')
# net1_left

# net2_right = train(hparams,path = root_files, face_side='right',network='net2',gpus=1, num_workers = num_workers)
# net2_right
# net2_left = train(hparams,path = root_files, face_side='left',network='net2')
# net2_left
