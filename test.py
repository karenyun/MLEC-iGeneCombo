import argparse
import collections
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import distutils.util


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
#     data_loader = config.init_obj('data_loader', module_data)
    data_loader = config.init_obj('test_data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    logger.info('the mode is {}'.format(config['mode']))
    logger.info('test cells are {}'.format(config['test_data_loader']['args']['test_cells']))
    logger.info('results are saved in {}'.format(config['saved_path']))

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    test_metrics = dict()
    output_list = []
    target_list = []

    # load fixed features
    fixed_edge_index = data_loader.dataset.network_input.to(device)
    fixed_node_feats = data_loader.dataset.network_node_feats.to(device)
    fixed_cell_node_feats = data_loader.dataset.network_cell_specific_node_feats.to(device)

    with torch.no_grad():
        for i, (inputs, target) in enumerate(tqdm(data_loader)):
#             print('---debug---')
#             print("i ", i)
#             print('inputs shape ', len(inputs))
            cell_input_exp, cell_input_ess = inputs[0].to(device), inputs[1].to(device)
            specific_omics_1, specific_omics_2 = inputs[2].to(device), inputs[3].to(device)
            gene1_idx, gene2_idx = inputs[4].to(device), inputs[5].to(device)
#             specific_omics = inputs[2].to(device)
#             gene1_idx, gene2_idx = inputs[3].to(device), inputs[4].to(device)

            inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)

            target = target.to(device)
            if config["test_data_loader"]["args"]["network_specific"]:
                output = model(inputs, fixed_edge_index, fixed_cell_node_feats)
            else:
                output = model(inputs, fixed_edge_index, fixed_node_feats)

            output_list.extend(output)
            target_list.extend(target)

        # computing loss, metrics on test set
        output_list = torch.tensor(output_list)
        target_list = torch.tensor(target_list)
        loss = loss_fn(output_list, target_list)
        
        test_metrics["loss"] = loss.item()
#         output_list = output_list * (3.932254644--6.027560721) + -6.027560721
        for met in metric_fns:
            test_metrics[ met.__name__ ] = met(output_list, target_list)

    log = {'loss': test_metrics["loss"]}
    log.update({
        met.__name__: test_metrics[met.__name__] for met in metric_fns
    })
    logger.info(log)

    print("saving results...")
    res_df = data_loader.get_original_file()
    res_df["target"] = target_list.tolist()
    res_df["pred"] = output_list.tolist()
    res_df.to_csv(config['saved_path'], index=False)
#     res_df.to_csv("data/res_df_tt_cell_22RV1_1.csv", index=False)
    


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
    args.add_argument('-p', '--saved_path', default="data/test.csv", type=str)
    args.add_argument('-v', '--test_mode', default=None, type=str)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--cell', '--cell_flag'], type=lambda x:bool(distutils.util.strtobool(x)), target='arch;args;cell_flag'),
        CustomArgs(['--network', '--network_flag'], type=lambda x:bool(distutils.util.strtobool(x)), target='arch;args;network_flag')
    ]
    config = ConfigParser.from_args(args, options)
#     config = ConfigParser.from_args(args)
    main(config)
