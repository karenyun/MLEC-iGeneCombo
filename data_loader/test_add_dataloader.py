from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
from pandas.errors import SettingWithCopyWarning
import json,random,os,sys,pickle
import warnings
import networkx as nx
import pickle
from sklearn.preprocessing import scale
from sklearn.metrics import matthews_corrcoef
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch_geometric.data import HeteroData

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class MyDataset(Dataset):
    def __init__(self, data_dir, input_file, CCLE_exp_file, CCLE_ess_file, CCLE_cn_file, cell_embeds_exp, cell_embeds_ess, network_type,
                                                                            gene_embeds_combined, network_node_embdings, task, thres, network_specific, cell_list=None, train_list=None, test_mode=None, kegg_genes_file=None):
        self.data_dir = data_dir
        self.input_file = input_file
        self.CCLE_exp_file = CCLE_exp_file
        self.CCLE_ess_file = CCLE_ess_file
        self.CCLE_cn_file = CCLE_cn_file
        self.cell_embeds_exp = cell_embeds_exp
        self.cell_embeds_ess = cell_embeds_ess
        self.network_type = network_type
        self.gene_embeds_combined = gene_embeds_combined
        self.network_node_embdings = network_node_embdings
        self.task = task
        self.thres = thres
        self.network_specific = network_specific
        self.train_list = train_list
        self.cell_list = cell_list
        self.test_mode = test_mode
        self.kegg_genes_file = kegg_genes_file
        self.test_df = pd.DataFrame(columns=["study_origin", "cell", "gene1", "gene2", "score", "sorted_pair"])

        self._load_data()
        

#         if cell_list is not None and len(cell_list) > 0:
#             print("In test mode...")
#             print("test mode is {}".format(test_mode))
#             self.effect_df_org = self.effect_df.copy(deep=True)
#             self.effect_df = self.effect_df[self.effect_df["cell"].isin(cell_list)]
# #             self.effect_df = self.effect_df[(self.effect_df["gene2"]=="AATF")|(self.effect_df["gene1"]=="AATF")]
# #             self.effect_df["score"] = self.effect_df["score"].apply(lambda x: (x--6.027560721)/(3.932254644--6.027560721))
#             if self.kegg_genes_file != None:
#                 unique_genes_specific_cell = list(set(self.effect_df["gene1"]) | set(self.effect_df["gene2"]))
#                 for pathway in self.kegg_pathways:
#                     kegg_genes = self.pathway_genes[pathway]
#                     intersaction_genes = list(set(unique_genes_specific_cell) & set(kegg_genes))
#                     if(len(intersaction_genes)>0):
#                         intersaction_gene_pairs = self.effect_df[(self.effect_df["gene1"].isin(intersaction_genes)) & (self.effect_df["gene2"].isin(intersaction_genes))]
#                         self.test_df = pd.concat([self.test_df, intersaction_gene_pairs], ignore_index=True)
#                         self.test_df.drop_duplicates(inplace=True)
# #                 self.effect_df = self.test_df
#                 print(self.test_df.shape)
#                 df = self.effect_df.merge(self.test_df, on=self.test_df.columns.to_list(), 
#                    how='left', indicator=True)
#                 self.effect_df = df.loc[df._merge=='left_only',df.columns!='_merge']
#                 print(self.effect_df.shape)


        self._filter_data()
        
        if cell_list is None:
            #####normalize the score
            min_score = np.min(self.effect_df["score"])
            max_score = np.max(self.effect_df["score"])
            ### log score 
            print("min_score is {}".format(min_score)) ### -6.02756072
            print("max_Score is {}".format(max_score)) ### 3.932254644
            self.effect_df["score"] = self.effect_df["score"].apply(lambda x: 2*(x-min_score)/(max_score-min_score)-1)
        else:
            print("org {}".format(self.effect_df["score"]))
            self.effect_df["score"] = self.effect_df["score"].apply(lambda x: 2*(x--6.066414806040367)/(4.84085277691662--6.066414806040367)-1)
            print(self.effect_df["score"])

        

        self._mapping_data()

        if "score" not in self.effect_df.columns:
            print("In prediction mode...")
            self.effect_df["score"] = -1

        self._generate_fixed_network_data()
        
        # if cell_list == None:
        
        #     self._generate_fixed_cell_network_data(self.train_list)
        # else:
        #     self._generate_fixed_cell_network_data(self.cell_list)
#         print("Final all cell line names are {}".format(self.effect_df["cell"].unique()))
#         print(self.effect_df["cell"].value_counts())  
#         unique_cells = self.effect_df_org['cell'].unique().tolist()
#         unique_cells = list(["OVCAR8","22RV1","786O","HT29","A375","K562","A549","JURKAT","MELJUSO"])
        unique_cells = list(["OVCAR8","22RV1","786O","HT29","A375","K562","A549","JURKAT","MELJUSO", "HS944T", "HS936T", "HSC5", "IPC298", "MEL202", "PATU8988S", "PK1", "GI1"])
#         if test_mode == "C1": 
        if cell_list is not None and len(cell_list) > 0:
            if test_mode != "ALL":
                self.effect_df = self._filter_gene_pairs(cell_list[0], unique_cells, test_mode)
#         else if test_mode == "C2":
#             effect_common_df = self._filter_gene_pairs(cell_list[0], unique_cells)
#             self.effect_df = pd.merge(self.effect_df_org, effect_common_df, indicator=True, how='outer').query("_merge == 'left_only'").drop('_merge', axis=1)
#         self.normalize_input_features()

        
    def _load_data(self):
        #print("loading mapping info...")
        #CCLE_info = pd.read_csv(self.data_dir + "omics/sample_info.csv")
        #self.CCLE_name2depmap = dict(zip(CCLE_info["stripped_cell_line_name"], CCLE_info["DepMap_ID"]))
        #self.CCLE_depmap2name= dict(zip(CCLE_info["DepMap_ID"], CCLE_info["stripped_cell_line_name"]))

        print("loading gene combination effect...")
        # columns are: "cell", "gene1", "gene2", "score"
        self.effect_df = pd.read_csv(self.data_dir + self.input_file)
        # self.effect_df = self.effect_df.rename(columns={"cell_line_origin":"cell", "sgRNA_target_1":"gene1", "sgRNA_target_2":"gene2", "Median LFC":"score"})
        self.unique_cells = self.effect_df["cell"].unique().tolist()
#         print("All cell line names are {}".format(self.effect_df["cell"].unique()))

        print("loading omics data...")
        self.CCLE_exp = pd.read_csv(self.data_dir + self.CCLE_exp_file, index_col = 0)
        self.CCLE_ess = pd.read_csv(self.data_dir + self.CCLE_ess_file, index_col = 0)
        # add cn
#         self.CCLE_cn = pd.read_csv(self.data_dir + self.CCLE_cn_file, index_col = 0)
#         print("CCLE cell line names are {}".format(self.CCLE_exp.index))

        self.cell_exp = pd.read_csv(self.data_dir + self.cell_embeds_exp, index_col = 0)
        self.cell_ess = pd.read_csv(self.data_dir + self.cell_embeds_ess, index_col = 0)

        self.network_node_feats_df = pd.read_csv(self.data_dir + self.gene_embeds_combined, index_col = 0)

        self.network_node_embeddings = pd.read_csv(self.data_dir + self.network_node_embdings, sep="\t", index_col = 0)#.reset_index().rename(columns={"index":"gene"})

        print("loading network data...")
        if self.network_type == "PPI":
            print("loading PPI...")
            network = pd.read_csv(self.data_dir + "networks/" + self.network_type + ".csv", index_col = 0)
            network = network[network['Experimental System Type'] == 'physical']
            network.rename(columns={'Official Symbol Interactor A':'gene1', 'Official Symbol Interactor B':'gene2'}, inplace=True)
            network = network[['gene1','gene2']]
            self.network = self.to_undirected(network)
        elif self.network_type == "pathway":
            print("loading pathway...")
            self.network = pd.read_csv(self.data_dir + "networks/" + self.network_type + ".csv")
            self.network.rename(columns={"Regulator":"gene1", "Target gene":"gene2"}, inplace=True)
            
        if self.kegg_genes_file != None:
            print("loading kegg genes data...")
            gene_kegg = pd.read_csv(self.data_dir + self.kegg_genes_file, index_col=0)
            self.kegg_pathways = list(set(gene_kegg["PathwayID"]))
            self.pathway_genes = dict()
            for pathway in self.kegg_pathways:   
                self.pathway_genes[pathway] = list(set(gene_kegg.loc[gene_kegg["PathwayID"]==pathway]["Symbol"]))

    
    def _filter_data(self):
        print("Filtering effect data...")
#         self.all_CCLE_genes = list(set(self.CCLE_exp.columns) & set(self.CCLE_ess.columns))
        # add cn
        self.all_CCLE_genes = list(set(self.CCLE_exp.index) & set(self.CCLE_ess.index)) #& set(self.CCLE_cn.columns))
        print(len(self.CCLE_exp.index))
        print(len(self.CCLE_ess.index))
        print(len(self.all_CCLE_genes))
        self.all_network_genes = list(set(self.network["gene1"].unique()) | set(self.network["gene2"].unique()))

        print("Original sample size is {}".format(self.effect_df.shape[0]))
        self.effect_df = self.effect_df[(self.effect_df["gene1"].isin(self.all_CCLE_genes))&(self.effect_df["gene2"].isin(self.all_CCLE_genes))]
        print("After filtering genes not in CCLE, the sample size is {}".format(self.effect_df.shape[0]))
        print("After filltering, all cell line names are {}".format(self.effect_df["cell"].unique()))

        self.all_genes = list(set(self.all_CCLE_genes) | set(self.all_network_genes))
    
    def _mapping_data(self):
        self.all_genes = sorted(self.all_genes)
        self.gene_mapping = dict(zip(self.all_genes, range(len(self.all_genes))))

        print("mapping genes into indexs in the PPI and pathway networks...")

        self.network["gene1"] = self.network["gene1"].apply(lambda x: self.gene_mapping[x])
        self.network["gene2"] = self.network["gene2"].apply(lambda x: self.gene_mapping[x])

        #self.network_node_feats = self.network_node_feats.loc[self.all_genes].values

        self.effect_df["gene1_idx"] = self.effect_df["gene1"].apply(lambda x: self.gene_mapping[x])
        self.effect_df["gene2_idx"] = self.effect_df["gene2"].apply(lambda x: self.gene_mapping[x])

    def _generate_fixed_network_data(self):
        print("Generating fixed network input data...")
        # process fixed features (rather than batch loading)
        self.network_input = torch.tensor([self.network["gene1"].values,
                                           self.network["gene2"].values],
                                           dtype=torch.long)
        print("network_input shape {}".format(self.network_input.shape))
        
        self.network_node_feats = np.zeros((len(self.all_genes), 384))#128))
        for gene, idx in self.gene_mapping.items():
            if gene in self.network_node_feats_df.index:
                self.network_node_feats[idx, :256] = self.network_node_feats_df.loc[gene]
            if gene in self.network_node_embeddings.index:
                self.network_node_feats[idx, 256:] = self.network_node_embeddings.loc[gene]
        
#         self.network_node_feats = scale(self.network_node_feats) #add normalization

        self.network_node_feats = torch.tensor(self.network_node_feats, dtype=torch.float)
        
    def _generate_fixed_cell_network_data(self, lists):
        print("Generating fixed cell line network input data...")
        # process fixed features (rather than batch loading)
#         self.network_cell_specific_input = self.network_input.unsqueeze(2).expand(-1,-1,len(lists))
        self.network_cell_specific_node_feats = np.zeros((len(lists), len(self.all_genes), 2))
        for i, cell in enumerate(lists):
            for gene, idx in self.gene_mapping.items():
                if gene in self.CCLE_ess.columns:
                    self.network_cell_specific_node_feats[i][idx][0] = self.CCLE_ess.loc[cell,gene]
                if gene in self.CCLE_exp.columns:
                    self.network_cell_specific_node_feats[i][idx][1] = self.CCLE_exp.loc[cell,gene]
        if len(lists)==1:
            self.network_cell_specific_node_feats = self.network_cell_specific_node_feats.reshape(-1,2)
        self.network_cell_specific_node_feats = torch.tensor(self.network_cell_specific_node_feats, dtype=torch.float)
            
    
    def normalize_input_features(self):
        print("Normalizing input features...")
#         self.network_node_feats = scale(self.network_node_feats)

        scaled_exp = scale(self.CCLE_exp.values, axis=1)
        self.CCLE_exp = pd.DataFrame(scaled_exp, index=self.CCLE_exp.index, columns=self.CCLE_exp.columns)

        scaled_ess = scale(self.CCLE_ess.values, axis=1)
        self.CCLE_ess = pd.DataFrame(scaled_ess, index=self.CCLE_ess.index, columns=self.CCLE_ess.columns)
    
    def _process_CCLE_file(self, df):
        # format gene names
        gene_names = list(map(lambda x:x.split(" ")[0], df.columns))
        df.columns = gene_names

        # change depmap ID to cell line names
        cell_names = list(map(lambda x:self.CCLE_depmap2name[x], df.index))
        df.index = cell_names

        return df

    def to_undirected(self, df):
        """
        make a directed graph into an undirected graph

        df: a dataframe where each row represents one edge from "gene1" to "gene2"
        """
        df_dup = df.reindex(columns=['gene2','gene1'])
        df_dup.columns = ['gene1','gene2']
        df = df._append(df_dup)
        df.drop_duplicates(inplace=True)

        return df
    
    def _filter_gene_pairs(self,cell_name, unique_cells, mode):
        print("----debug----")
        print("cell_name {}".format(cell_name))
        print("unique_cells {}".format(unique_cells))
        left_cells = list(set(unique_cells) - set([cell_name]))
        print("left cells {}".format(left_cells))
        left_gene_comb = self.effect_df_org[self.effect_df_org["cell"].isin(left_cells)]
#         if mode == "C1":
        left_gene_pairs = left_gene_comb[['gene1', 'gene2']]
        left_gene_pairs_dup = left_gene_pairs.reindex(columns=['gene2', 'gene1'])
        left_gene_pairs_dup.columns = ['gene1', 'gene2']
        left_gene_pairs._append(left_gene_pairs_dup)
        left_gene_pairs.drop_duplicates(inplace=True)

#         main_gene_pairs = self.effect_df.loc[self.effect_df['cell'] == cell_name, ['gene1', 'gene2']]
        
        merged_df = pd.merge(self.effect_df, left_gene_pairs, how='inner', on=['gene1', 'gene2'])
        print(merged_df.shape[0])
        if mode == "C1":
            return merged_df
#         print(set(merged_df["gene1"].unique().tolist()+merged_df["gene2"].unique().tolist()))
        genes = list(set(left_gene_comb["gene1"].unique()) | set(left_gene_comb["gene2"].unique()))
#         print(genes)
        if mode == "C2":            
            common_df = self.effect_df[self.effect_df["gene1"].isin(genes) | self.effect_df["gene2"].isin(genes)]
            merged_df = pd.merge(common_df, merged_df, indicator=True, how='outer').query("_merge == 'left_only'").drop('_merge', axis=1)
        elif mode == "C3":
            merged_df = self.effect_df[~self.effect_df["gene1"].isin(genes) & ~self.effect_df["gene2"].isin(genes)]
            
        
        print(merged_df.shape[0])

        return merged_df


    def __len__(self):
        # columns: cell, gene1, gene2, gene1_idx, gene2_idx, score
        return self.effect_df.shape[0]

    def __getitem__(self, idx):
        cell = self.effect_df.iloc[idx]["cell"]
        # print(cell, type(cell))
#         if cell == 'RPE1':
#             print('cell name ',cell)
        # print(f"------1------")
        gene1, gene1_idx = self.effect_df.iloc[idx]["gene1"], self.effect_df.iloc[idx]["gene1_idx"]
        gene2, gene2_idx = self.effect_df.iloc[idx]["gene2"], self.effect_df.iloc[idx]["gene2_idx"]
        # print(f"------2------")
        cell_input_exp = torch.tensor(self.cell_exp.loc[cell].to_numpy(), dtype=torch.float)
        cell_input_ess = torch.tensor(self.cell_ess.loc[cell].to_numpy(), dtype=torch.float)
        # print(f"------3------")
        specific_omics_1 = torch.tensor([self.CCLE_exp.loc[gene1, cell], self.CCLE_ess.loc[gene1, cell]], dtype=torch.float)
        specific_omics_2 = torch.tensor([self.CCLE_exp.loc[gene2, cell], self.CCLE_ess.loc[gene2, cell]], dtype=torch.float)
        # print(f"------4------")
#         specific_omics_1 = torch.tensor([self.CCLE_exp.loc[cell, gene1], self.CCLE_ess.loc[cell, gene1]self.CCLE_exp.loc[cell, gene1]**2, self.CCLE_ess.loc[cell, gene1]**2], dtype=torch.float)
#         specific_omics_2 = torch.tensor([self.CCLE_exp.loc[cell, gene2], self.CCLE_ess.loc[cell, gene2], self.CCLE_exp.loc[cell, gene2]**2, self.CCLE_ess.loc[cell, gene2]**2], dtype=torch.float)
#         specific_omics_1 = torch.tensor([self.CCLE_exp.loc[cell, gene1], self.CCLE_ess.loc[cell, gene1], self.CCLE_cn.loc[cell, gene1]], dtype=torch.float)
#         specific_omics_2 = torch.tensor([self.CCLE_exp.loc[cell, gene2], self.CCLE_ess.loc[cell, gene2], self.CCLE_cn.loc[cell, gene2]], dtype=torch.float)
        inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, 
                                    torch.tensor(gene1_idx, dtype=torch.long), torch.tensor(gene2_idx, dtype=torch.long))
        
        
        if self.task == "regression":
            target = torch.tensor(self.effect_df.iloc[idx]["score"], dtype=torch.float)
        elif self.task == "classification":
            target = torch.tensor(self.effect_df.iloc[idx]["score"] < self.thres, dtype=torch.float)
        
        # if self.network_specific and self.cell_list == None:  ##training
        #     cell_index = self.train_list.index(cell)
        #     inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, 
        #                             torch.tensor(gene1_idx, dtype=torch.long), torch.tensor(gene2_idx, dtype=torch.long), torch.tensor(cell_index, dtype=torch.long))
            
        #     return inputs, target
        '''
        if self.network_specific:
#             print("data loader, network specific")
            network_node_feats = np.zeros((len(self.all_genes), 2))
            for gene, idx in self.gene_mapping.items():
                if gene in self.CCLE_ess.columns:
                    network_node_feats[idx][0] = self.CCLE_ess.loc[cell,gene]
                if gene in self.CCLE_exp.columns:
                    network_node_feats[idx][1] = self.CCLE_exp.loc[cell,gene]
            #self.network_node_feats = scale(self.network_node_feats)

            network_node_feats = torch.tensor(network_node_feats, dtype=torch.float)
            inputs = (cell_input_exp, cell_input_ess, specific_omics_1, specific_omics_2, 
                                    torch.tensor(gene1_idx, dtype=torch.long), torch.tensor(gene2_idx, dtype=torch.long), network_node_feats)
            return inputs, target
        '''

        return inputs, target


class DataLoaderPredict(DataLoader):
    """
    data loader for predicting new samples because other implementations contains shuffling of samples
    use SequentialSampler instead
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, split_method, num_workers, input_file, test_cells, CCLE_exp_file, 
                                                CCLE_ess_file, CCLE_cn_file, cell_embeds_exp, cell_embeds_ess, kegg_genes_file, network_type, gene_embeds_combined, network_node_embdings, task, thres, network_specific, test_mode):
        #debug
#         cell_list = list(["OVCAR8","22RV1","786O","HT29","A375","K562","A549","JURKAT","MELJUSO"])
#         cell_list = list(["22RV1"])
        cell_list = test_cells
        if kegg_genes_file == []:
            kegg_genes_file = None
        self.dataset = MyDataset(data_dir, input_file, CCLE_exp_file, CCLE_ess_file, CCLE_cn_file, cell_embeds_exp, cell_embeds_ess, network_type,
                                                                                    gene_embeds_combined, network_node_embdings, task, thres, network_specific, cell_list, test_mode=test_mode, kegg_genes_file=kegg_genes_file)
        self.task = task

        self.sampler = SequentialSampler(self.dataset)
        
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers
        }

        super().__init__(sampler=self.sampler, **self.init_kwargs)
    
    def get_original_file(self):
        return self.dataset.effect_df.drop(columns=["score","gene1_idx","gene2_idx"])
