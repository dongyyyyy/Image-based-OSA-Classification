import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

class two_views_MHSA(nn.Module):
    def __init__(self, backbone1: nn.Module, backbone2: nn.Module, backbone_freezing: bool=True,
                 shared_attention:bool=False,
                    backbone_output_dim: int=2048,emb_size: int=768, num_patches: int=196,depth: int=1,
                    positional_embedding: bool=True,cls_token: bool=False,last_gelu=False,
                    heads: int = 8,dropout:float=0.,emb_dropout:float=0., class_num=2,last_mlps=False):
        super(two_views_MHSA, self).__init__()
        self.heads = heads
        # ResNet-50 (input :448)
        self.backbone1 = backbone1 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone2 = backbone2 # [B, 2048, 14, 14] Batch, channel, height, width
        # if backbone3 is not None:
        #     self.backbone3 = backbone3

        # Freezing
        if backbone_freezing:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
            # if backbone3 is not None:
            #     for param in self.backbone3.parameters():
            #         param.requires_grad = False
        if backbone_output_dim != emb_size: # emb_size = MHSA 
            self.fc_backbone1 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone2 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
        else:
            self.fc_backbone1 = nn.Identity()
            self.fc_backbone2 = nn.Identity()
            
        self.positional_embedding = positional_embedding
        self.cls_token = cls_token
        
        if self.cls_token:
            self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_size))
            # if self.backbone3 is not None:
            #     self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            num_patches = num_patches + 1
        if self.positional_embedding:
            self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        else:
            print('Do not use Pos Embedding!!')
            # if self.backbone3 is not None:
            #     self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        self.dropout = nn.Dropout(emb_dropout)

        # Cross Attention
        self.mha_1 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                             dropout=dropout, batch_first=True)
        self.mha_2 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                            dropout=dropout, batch_first=True)
        
        self.concat_linear = nn.Linear(in_features=2 * emb_size, out_features= emb_size)
        self.fc = nn.Linear(in_features= emb_size, out_features=class_num)




    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out1 = self.backbone1(x) # [batch,2048,7,7]
        out2 = self.backbone2(y) # [batch,2048,7,7]
        

        batch_out1, emb_size_out1, height_out1, width_out1 = out1.size()
        batch_out2, emb_size_out2, height_out2, width_out2 = out2.size()

        out1 = out1.reshape(batch_out1,emb_size_out1,width_out1*height_out1)
        out2 = out2.reshape(batch_out2,emb_size_out2,width_out2*height_out2)

        # [Batch, Channel, Patch] -> [Batch, Patch, Channel]
        out1 = out1.permute(0,2,1)
        out2 = out2.permute(0,2,1)


        out1 = self.fc_backbone1(out1) # [batch, 196, emb_size]
        out2 = self.fc_backbone2(out2) # [batch, 196, emb_size]

        if self.cls_token:
            cls_token_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = batch_out1)
            cls_token_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = batch_out2)

            out1 = torch.cat((cls_token_1,out1),dim=1)
            out2 = torch.cat((cls_token_2,out2),dim=1)
            

        # Adding Positional Embedding
        if self.positional_embedding:
            # print(f'out1 shape = {out1.shape} // self.pos_embedding_1 shape = {self.pos_embedding_1.shape}')
            # exit(1)
            out1 = out1 + self.pos_embedding_1
            out2 = out2 + self.pos_embedding_2
        
        # dropout
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

        out1,_ = self.mha_1(out1,out1,out1) 
        out2,_ = self.mha_2(out2,out2,out2) 

        out1_mean = torch.mean(out1, dim=1)
        out2_mean = torch.mean(out2, dim=1)

        mean_stacked = torch.stack((out1_mean, out2_mean), dim=1) 
        static_mean , static_std = torch.std_mean(mean_stacked, dim=1)

        out = torch.cat((static_mean,static_std),dim=1)
        out = self.concat_linear(out)
        out = self.fc(out)
              
        return out
    
class two_views_MHCA(nn.Module):
    def __init__(self, backbone1: nn.Module, backbone2: nn.Module, backbone_freezing: bool=True,
                 shared_attention:bool=False,
                    backbone_output_dim: int=2048,emb_size: int=768, num_patches: int=196,depth: int=1,
                    positional_embedding: bool=True,cls_token: bool=False,last_gelu=False,
                    heads: int = 8,dropout:float=0.,emb_dropout:float=0., class_num=2,last_mlps=False):
        super(two_views_MHCA, self).__init__()
        self.heads = heads
        # ResNet-50 (input :448)
        self.backbone1 = backbone1 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone2 = backbone2 # [B, 2048, 14, 14] Batch, channel, height, width
        # if backbone3 is not None:
        #     self.backbone3 = backbone3

        # Freezing
        if backbone_freezing:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
            # if backbone3 is not None:
            #     for param in self.backbone3.parameters():
            #         param.requires_grad = False
        if backbone_output_dim != emb_size: # emb_size = MHSA 
            self.fc_backbone1 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone2 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
        else:
            self.fc_backbone1 = nn.Identity()
            self.fc_backbone2 = nn.Identity()
            
        self.positional_embedding = positional_embedding
        self.cls_token = cls_token
        
        if self.cls_token:
            self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_size))
            # if self.backbone3 is not None:
            #     self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            num_patches = num_patches + 1
        if self.positional_embedding:
            self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        else:
            print('Do not use Pos Embedding!!')
            # if self.backbone3 is not None:
            #     self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        self.dropout = nn.Dropout(emb_dropout)

        # Cross Attention
        self.mhca_1 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                             dropout=dropout, batch_first=True)
        self.mhca_2 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                             dropout=dropout, batch_first=True)
        
        self.concat_linear = nn.Linear(in_features=2 * emb_size, out_features= emb_size)
        self.fc = nn.Linear(in_features= emb_size, out_features=class_num)




    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out1 = self.backbone1(x) # [batch,2048,7,7]
        out2 = self.backbone2(y) # [batch,2048,7,7]
        

        batch_out1, emb_size_out1, height_out1, width_out1 = out1.size()
        batch_out2, emb_size_out2, height_out2, width_out2 = out2.size()
        

        out1 = out1.reshape(batch_out1,emb_size_out1,width_out1*height_out1)
        out2 = out2.reshape(batch_out2,emb_size_out2,width_out2*height_out2)
        

        # [Batch, Channel, Patch] -> [Batch, Patch, Channel]
        out1 = out1.permute(0,2,1)
        out2 = out2.permute(0,2,1)

        out1 = self.fc_backbone1(out1) # [batch, 196, emb_size]
        out2 = self.fc_backbone2(out2) # [batch, 196, emb_size]

        if self.cls_token:
            cls_token_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = batch_out1)
            cls_token_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = batch_out2)

            out1 = torch.cat((cls_token_1,out1),dim=1)
            out2 = torch.cat((cls_token_2,out2),dim=1)

        # Adding Positional Embedding
        if self.positional_embedding:
            # print(f'out1 shape = {out1.shape} // self.pos_embedding_1 shape = {self.pos_embedding_1.shape}')
            # exit(1)
            out1 = out1 + self.pos_embedding_1
            out2 = out2 + self.pos_embedding_2
        
        # dropout
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

        out1,_ = self.mhca_1(out2,out1,out1) 
        out2,_ = self.mhca_2(out1,out2,out2) 
        

        out1_mean = torch.mean(out1, dim=1)
        out2_mean = torch.mean(out2, dim=1)
        

        mean_stacked = torch.stack((out1_mean,out2_mean), dim=1) 
        static_mean , static_std = torch.std_mean(mean_stacked, dim=1)

        out = torch.cat((static_mean,static_std),dim=1)
        out = self.concat_linear(out)
        out = self.fc(out)
              
        return out
    

class two_views_MHDA(nn.Module):
    def __init__(self, backbone1: nn.Module, backbone2: nn.Module, backbone_freezing: bool=True,
                 shared_attention:bool=False,
                    backbone_output_dim: int=2048,emb_size: int=768, num_patches: int=196,depth: int=1,
                    positional_embedding: bool=True,cls_token: bool=False,last_gelu=False,
                    heads: int = 8,dropout:float=0.,emb_dropout:float=0., class_num=2,last_mlps=False):
        super(two_views_MHDA, self).__init__()
        self.heads = heads
        # ResNet-50 (input :448)
        self.backbone1 = backbone1 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone2 = backbone2 # [B, 2048, 14, 14] Batch, channel, height, width
        # if backbone3 is not None:
        #     self.backbone3 = backbone3

        # Freezing
        if backbone_freezing:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
            
            
        if backbone_output_dim != emb_size: # emb_size = MHSA 
            self.fc_backbone1 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone2 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            
        else:
            self.fc_backbone1 = nn.Identity()
            self.fc_backbone2 = nn.Identity()
            
            
        self.positional_embedding = positional_embedding
        self.cls_token = cls_token
        
        if self.cls_token:
            self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_size))
            
            num_patches = num_patches + 1
        if self.positional_embedding:
            self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            
        else:
            print('Do not use Pos Embedding!!')
            # if self.backbone3 is not None:
            #     self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        self.dropout = nn.Dropout(emb_dropout)

        # Cross Attention
        self.mhca_1 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                             dropout=dropout, batch_first=True)
        self.mhca_2 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                             dropout=dropout, batch_first=True)
        
        # Self Attention
        self.mhsa_1 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                             dropout=dropout, batch_first=True)
        self.mhsa_2 = nn.MultiheadAttention(embed_dim=emb_size, num_heads=heads,
                                            dropout=dropout, batch_first=True)

        self.concat_linear = nn.Linear(in_features=2 * emb_size, out_features= emb_size)
        self.fc = nn.Linear(in_features= emb_size, out_features=class_num)




    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        out1 = self.backbone1(x) # [batch,2048,7,7]
        out2 = self.backbone2(y) # [batch,2048,7,7]

        batch_out1, emb_size_out1, height_out1, width_out1 = out1.size()
        batch_out2, emb_size_out2, height_out2, width_out2 = out2.size()

        out1 = out1.reshape(batch_out1,emb_size_out1,width_out1*height_out1)
        out2 = out2.reshape(batch_out2,emb_size_out2,width_out2*height_out2)

        # [Batch, Channel, Patch] -> [Batch, Patch, Channel]
        out1 = out1.permute(0,2,1)
        out2 = out2.permute(0,2,1)

        out1 = self.fc_backbone1(out1) # [batch, 196, emb_size]
        out2 = self.fc_backbone2(out2) # [batch, 196, emb_size]

        if self.cls_token:
            cls_token_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = batch_out1)
            cls_token_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = batch_out2)

            out1 = torch.cat((cls_token_1,out1),dim=1)
            out2 = torch.cat((cls_token_2,out2),dim=1)

        # Adding Positional Embedding
        if self.positional_embedding:
            # print(f'out1 shape = {out1.shape} // self.pos_embedding_1 shape = {self.pos_embedding_1.shape}')
            # exit(1)
            out1 = out1 + self.pos_embedding_1
            out2 = out2 + self.pos_embedding_2
        
        # dropout
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

        out1_ca,_ = self.mhsa_1(out2,out1,out1) 
        out2_ca,_ = self.mhsa_2(out1,out2,out2) 

        out1_sa,_ = self.mhca_1(out1,out1,out1) 
        out2_sa,_ = self.mhca_2(out2,out2,out2) 

        

        out1_sa_mean = torch.mean(out1_sa, dim=1)
        out2_sa_mean = torch.mean(out2_sa, dim=1)
        
        out1_ca_mean = torch.mean(out1_ca, dim=1)
        out2_ca_mean = torch.mean(out2_ca, dim=1)
        
        

        mean_stacked = torch.stack((out1_sa_mean,out2_sa_mean,out1_ca_mean,out2_ca_mean), dim=1) 
        static_mean , static_std = torch.std_mean(mean_stacked, dim=1)

        out = torch.cat((static_mean,static_std),dim=1)
        out = self.concat_linear(out)
        out = self.fc(out)
              
        return out