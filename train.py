import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import distutils.util


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
CUDA_LAUNCH_BLOCKING = 1

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    print("Number of training batches: ", len(data_loader))
    print("Number of validation batches: ", len(valid_data_loader))
    print("cell {}".format(config['arch']['args']['cell_flag']))
    print('gene {}'.format(config['arch']['args']['network_flag']))
    print('lr {}'.format(config['optimizer']['args']['lr']))
    logger.info(config['arch']['args']['cell_flag'])
    logger.info(config['arch']['args']['network_flag'])
    logger.info(config['data_loader']['args']['test_cells'])
    logger.info(config['trainer']['epochs'])
#     logger.info(config)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    
    print("------------------------------------")
    print("Start training...")
    trainer.train()

    #print("------------------------------------")
    #print("Loading test data...")
    #test_data_loader = config.init_obj('test_data_loader', module_data)

    #print("Start testing on test set...")
    #trainer.test(test_data_loader)


if __name__ == '__main__':
    print(f"---debug---{torch.cuda.is_available()},---gpus---{torch.cuda.device_count()}")
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-t', '--test_cells', default=[], nargs='+')
    args.add_argument('-m', '--mode', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--step', '--step_size'], type=int, target='lr_scheduler;args;step_size'),
        CustomArgs(['--cell', '--cell_flag'], type=lambda x:bool(distutils.util.strtobool(x)), target='arch;args;cell_flag'),
        CustomArgs(['--network', '--network_flag'], type=lambda x:bool(distutils.util.strtobool(x)), target='arch;args;network_flag'),
        CustomArgs(['--epochs'], type=int, target='trainer;epochs')
    ]
    
    config = ConfigParser.from_args(args, options)
#     print("cell {}".format(config['arch']['args']['cell_flag']))
#     print('gene {}'.format(config['arch']['args']['network_flag']))
#     print('lr {}'.format(config['optimizer']['args']['lr']))
    main(config)
