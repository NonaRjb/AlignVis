import sys

sys.path.append("/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img")

import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict
from braindecode.models import EEGConformer
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from src.brain_architectures import EEGNet, NICE, BrainMLP, ResNet1d, lstm


class BrainEncoder(nn.Module): # TODO: now every architecture has a classification layer -> embedding should be extracted from penultimate layer.
    def __init__(self,
                 embed_dim=1024,
                 backbone="eegnet",
                 n_channels: int = 128,
                 n_samples: int = 512,
                 n_classes: int = 40,
                 model_path: str = None,
                 **kwargs
                 ):
        super(BrainEncoder, self).__init__()

        device = kwargs["device"]

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.backbone_type = backbone
        self.embed_dim = embed_dim
        self.checkpoint = torch.load(model_path)['model_state_dict'] if model_path is not None else None
        
        if self.checkpoint is not None:
            new_state_dict = {}
            for key, value in self.checkpoint.items():
                # Remove 'brain_backbone.' from the keys
                new_key = key.replace("brain_backbone.", "")
                new_state_dict[new_key] = value
            
            self.checkpoint = new_state_dict

        if backbone == 'eegnet':
            self.brain_backbone = EEGNet(n_samples=n_samples, n_channels=n_channels, n_classes=n_classes) 
            if self.checkpoint:
                self.brain_backbone.load_state_dict(self.checkpoint) 
            self.feature_dim = list(self.brain_backbone.model.children())[-1].in_features
            self.return_node = 'model.do2'
            # self.brain_backbone = nn.Sequential(*list(self.brain_backbone.model.children())[:-1])
            print(get_graph_node_names(self.brain_backbone))

        elif backbone == 'nice':
            print("Using NICE backbone")
            self.brain_backbone = NICE(emb_size=kwargs['emb_size'], embedding_dim=kwargs['embedding_dim'], proj_dim=embed_dim) # 217360 for MEG and 1440 for EEG
            if self.checkpoint:
                self.brain_backbone.load_state_dict(self.checkpoint)
            self.feature_dim = embed_dim
            print(get_graph_node_names(self.brain_backbone))
            self.return_node = 'projector.2'
        
        elif backbone == 'eegconformer':
            self.brain_backbone = EEGConformer(
                n_outputs=None, 
                n_chans=n_channels, 
                n_filters_time=40, 
                filter_time_length=10, 
                pool_time_length=25, 
                pool_time_stride=5, 
                drop_prob=0.25, 
                att_depth=2, 
                att_heads=1, 
                att_drop_prob=0.5, 
                final_fc_length=1760, 
                return_features=False, 
                n_times=None, 
                chs_info=None, 
                input_window_seconds=None, 
                n_classes=1024, # fixed embedding size 
                input_window_samples=n_samples, 
                add_log_softmax=True)
            self.feature_dim = 1024
            # print(get_graph_node_names(self.brain_backbone))
        
        # elif backbone == 'atms':
        #     print("Using ATMS backbone")
        #     self.brain_backbone = eeg_architectures.ATMS(num_latents=embed_dim)
        #     if self.checkpoint:
        #         self.brain_backbone.load_state_dict(self.checkpoint)
        #     self.feature_dim = embed_dim
        #     self.return_node = 'projector.2'
        
        elif backbone == 'lstm':
            self.brain_backbone = lstm(input_size=n_channels, lstm_size=kwargs['lstm_size'],
                                           lstm_layers=kwargs['lstm_layers'], device=device)
            print(get_graph_node_names(self.brain_backbone))
        elif backbone == 'brain-mlp':
            self.brain_backbone = BrainMLP(out_dim=embed_dim, in_dim=int(n_channels*n_samples), clip_size=embed_dim)
            if self.checkpoint:
                self.brain_backbone.load_state_dict(self.checkpoint) 
            self.feature_dim = embed_dim
            self.return_node = 'projector.8'
            print(get_graph_node_names(self.brain_backbone))
        elif backbone == 'resnet1d':
            net_filter_size = kwargs['net_filter_size'] if 'net_filter_size' in kwargs.keys() else [64, 128, 196, 256, 320] # [16, 16, 32, 32, 64]
            net_seq_length = kwargs['net_seq_length'] if 'net_seq_length' in kwargs.keys() else [n_samples, 128, 64, 32, 16]
            self.brain_backbone = ResNet1d(
                n_channels=n_channels, 
                n_samples=n_samples, 
                net_filter_size=net_filter_size, 
                # net_seq_length=[n_samples, 128, 64, 32, 16],
                net_seq_length=net_seq_length, 
                n_classes=n_classes)
            if self.checkpoint:
                self.brain_backbone.load_state_dict(self.checkpoint) 
            self.feature_dim = list(self.brain_backbone.children())[-1].in_features
            self.return_node = 'view'
            print(get_graph_node_names(self.brain_backbone))
        else:
            raise NotImplementedError
        
        if 'subj' not in backbone and 'atm' not in backbone and 'eegconformer' not in backbone:
            self.brain_backbone = create_feature_extractor(self.brain_backbone, return_nodes=[self.return_node])
        print("feature dim = ", self.feature_dim)
        
        self.repr_layer = torch.nn.Linear(self.feature_dim, embed_dim)

    def forward(self, x, subject_id=None):
    
        if "resnet1d" in self.backbone_type or 'atm' in self.backbone_type or 'eegconformer' in self.backbone_type:
            x = x.squeeze(1)
        if "subj" in self.backbone_type or 'atm' in self.backbone_type:
            out = self.brain_backbone(x, subject_id)
        elif "eegconformer" in self.backbone_type:
            out = self.brain_backbone(x)
        else:
            out = self.brain_backbone(x)[self.return_node]
        out = out.view(out.size(0), -1)
        
        embedding = self.repr_layer(out)
        
        return embedding.squeeze(dim=1)


class SubjLinear(nn.Module):
    def __init__(self, input_dim, output_dim, subject_ids):
        super().__init__()
        self.comm_lin = nn.Linear(input_dim, output_dim)
        self.lin = nn.ModuleDict({
            str(subj_id): nn.Sequential(nn.Dropout(0.5), nn.Linear(output_dim, output_dim)) for subj_id in subject_ids
        })
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, subj_id):
        
        x = self.comm_lin(x)
    
        if isinstance(subj_id, list):
            x = [self.lin[str(id)](x_i) for id, x_i in zip(subj_id, x)]
            x = torch.stack(x)  # Stack back into a tensor after processing each element
        else:
            x = self.lin[str(subj_id)](x)

        return x
    
    def add_subject(self, subj_id):
        # Check if the subject already exists
        if subj_id in self.lin.keys():
            print(f"Subject {subj_id} already exists!")
        else:
            # Add a new Conv1d + BatchNorm1d module for the new subject
            self.lin.update({str(subj_id): nn.Sequential(nn.Dropout(0.25), nn.Linear(self.output_dim, self.output_dim))})
            print(f"Subject {subj_id} added successfully!")



class MLPHead(nn.Module):
    def __init__(
        self,
        input_size,
        n_classes,
        n_layers,
        hidden_size,
        **kwargs
        ):

        super().__init__()

        if n_layers == 1:
            self.mlp = torch.nn.Linear(input_size, n_classes)
        else:
            mlp_head_list = []
            feature_dim = input_dim
            for i in range(n_layers - 1):
                mlp_head_list.append(('ln' + str(i+1), torch.nn.Linear(feature_dim, hidden_dim)))
                mlp_head_list.append(('bn' + str(i+1), torch.nn.BatchNorm1d(hidden_dim))),
                mlp_head_list.append(('relu' + str(i+1), torch.nn.ReLU())),
                feature_dim = hidden_dim
            mlp_head_list.append(('lnout', torch.nn.Linear(hidden_dim, n_classes)))
            self.mlp = torch.nn.Sequential(OrderedDict(mlp_head_list))
    
    def forward(self, x):
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    path_to_chkpnt = "/proj/rep-learning-robotics/users/x_nonra/data/asif_out/spampinato_eegnet_2024-09-16_11-21-04/eegnet_spampinato.pth"
    model = EEGEncoder(backbone="brain-mlp", embed_dim=768, n_channels=17, n_samples=128, device="cuda" if torch.cuda.is_available() else "cpu")
    x_in = torch.randn((1, 1, 17, 128))
    x_out = model(x_in)
    print(x_out.shape)

    