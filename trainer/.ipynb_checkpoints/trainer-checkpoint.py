import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])#, writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])#, writer=self.writer)

        # load fixed features
        self.fixed_edge_index = self.data_loader.dataset.network_input.to(self.device)
        self.fixed_node_feats = self.data_loader.dataset.network_node_feats.to(self.device)
#         self.fixed_cell_edge_index = self.data_loader.dataset.network_cell_specific_input.to(self.device)
        self.fixed_cell_specific_node_feats = self.data_loader.dataset.network_cell_specific_node_feats.to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
#         print("debug, epoch {}".format(epoch))
        self.model.train()
        self.train_metrics.reset()

        # store the results from each batch into the list, used for evaluation
        output_list = []
        target_list = []
        loss_list = []

        for batch_idx, (inputs, target) in enumerate(self.data_loader):
#             print('batch_idx {}'.format(batch_idx))
#             self.model.half()
            cell_input_exp, cell_input_ess = inputs[0].to(self.device), inputs[1].to(self.device)
            specific_omics_1, specific_omics_2 = inputs[2].to(self.device), inputs[3].to(self.device)
            gene1_idx, gene2_idx = inputs[4].to(self.device), inputs[5].to(self.device)
#             print("original input len {}".format(len(inputs)))
            
            
            target = target.to(self.device)

            self.optimizer.zero_grad()
            if self.config["data_loader"]["args"]["network_specific"]:
#                 print("trainer, network specific")
                cell_index = inputs[6].to(self.device)
                inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)
                output = self.model(inputs, self.fixed_edge_index, self.fixed_cell_specific_node_feats, cell_index=cell_index)
#                 print("------end------")
            else:
#                 print("------debug------")
                inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)
                output = self.model(inputs, self.fixed_edge_index, self.fixed_node_feats)
#             print("start loss")
            loss = self.criterion(output, target)
            loss.backward()
#             self.model.float()
            self.optimizer.step()

            loss_list.append(loss.item())
#             print('loss is ', loss.item())
            output_list.extend(output)
            target_list.extend(target)

            if batch_idx == self.len_epoch:
                break
        
        #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.train_metrics.update('loss', np.mean(loss_list), epoch)
        self.writer.add_scalar('loss', np.mean(loss_list),epoch)
        self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'],epoch)
#         self.writer.add_scalar('lr', )
        output_list = torch.tensor(output_list)
        target_list = torch.tensor(target_list)

        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(output_list, target_list), epoch)
            self.writer.add_scalar(met.__name__, met(output_list, target_list), epoch)
        # get a dictionary storing evaluation results on training set
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        with torch.no_grad():
            output_list = []
            target_list = []
            for batch_idx, (inputs, target) in enumerate(self.valid_data_loader):
                cell_input_exp, cell_input_ess = inputs[0].to(self.device), inputs[1].to(self.device)
                specific_omics_1, specific_omics_2 = inputs[2].to(self.device), inputs[3].to(self.device)
                gene1_idx, gene2_idx = inputs[4].to(self.device), inputs[5].to(self.device)

#                 inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)
                target = target.to(self.device)
                
                if self.config["data_loader"]["args"]["network_specific"]:
#                     print("trainer, network specific")
                    cell_index = inputs[6].to(self.device)
                    inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)
                    output = self.model(inputs, self.fixed_edge_index, self.fixed_cell_specific_node_feats, cell_index=cell_index)
#                     print("------end------")
                else:
                    inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)
                    output = self.model(inputs, self.fixed_edge_index, self.fixed_node_feats)
                output_list.extend(output)
                target_list.extend(target)

            output_list = torch.tensor(output_list)
            target_list = torch.tensor(target_list) 
            loss = self.criterion(output_list, target_list)

            #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item(), epoch)
            self.writer.add_scalar('val_loss', loss.item(), epoch)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(output_list, target_list), epoch)
                self.writer.add_scalar('val_'+met.__name__, met(output_list, target_list), epoch)

        # add histogram of model parameters to the tensorboard
#         for name, p in self.model.named_parameters():
#             self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def test(self, test_data_loader):
        best_model_path = str(self.checkpoint_dir / 'model_best.pth')
        print("loading best model checkpoint {}...".format(best_model_path))
        checkpoint = torch.load(best_model_path)
        state_dict = checkpoint['state_dict']
        model = self.model
        model.load_state_dict(state_dict)

        test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        model.eval()
        with torch.no_grad():
            output_list = []
            target_list = []

            for batch_idx, (inputs, target) in enumerate(test_data_loader):
                cell_input_exp, cell_input_ess = inputs[0].to(self.device), inputs[1].to(self.device)
                specific_omics_1, specific_omics_2 = inputs[2].to(self.device), inputs[3].to(self.device)
                gene1_idx, gene2_idx = inputs[4].to(self.device), inputs[5].to(self.device)

                inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, gene1_idx, gene2_idx)
                target = target.to(self.device)
                # note that fixed features can be shared between test dataloader and train dataloader
                output = model(inputs, self.fixed_edge_index, self.fixed_node_feats)

                output_list.extend(output)
                target_list.extend(target)

            output_list = torch.tensor(output_list)
            target_list = torch.tensor(target_list) 
            loss = self.criterion(output_list, target_list)

            test_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                test_metrics.update(met.__name__, met(output_list, target_list),0)
        
        print(test_metrics.result())

        save_path = str(self.checkpoint_dir / 'prediction_res.csv')
        print("saving prediction results to {}...".format(save_path))
        res_df = test_data_loader.get_original_file()
        res_df["target"] = target_list.tolist()
        res_df["pred"] = output_list.tolist()
        
        res_df.to_csv(save_path, index=False)