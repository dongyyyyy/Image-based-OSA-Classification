from . import *
from ..Attention_modules import *

from model.Attention.three_inputs.Attention_modules import Encoder_blocks_three_inputs, Encoder_blocks_three_inputs_MHCA
    
class TF_Attention(nn.Module):
    def __init__(self, backbone1: nn.Module, backbone2: nn.Module,backbone3: nn.Module, backbone_freezing: bool=True,
                 shared_attention:bool=False,
                    backbone_output_dim: int=2048,emb_size: int=768, num_patches: int=196,depth: int=1,
                    positional_embedding: bool=True,cls_token: bool=False,last_gelu=False,
                    heads: int = 8,dropout:float=0.,emb_dropout:float=0., class_num=2):
# backbone3: nn.Module=None,
        super(TF_Attention, self).__init__()
        self.heads = heads
        # ResNet-50 (input :448)
        self.backbone1 = backbone1 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone2 = backbone2 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone3 = backbone3
        # if backbone3 is not None:
        #     self.backbone3 = backbone3

        # Freezing
        if backbone_freezing:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
            for param in self.backbone3.parameters():
                param.requires_grad = False
            # if backbone3 is not None:
            #     for param in self.backbone3.parameters():
            #         param.requires_grad = False
        if backbone_output_dim != emb_size: # emb_size = MHSA 
            self.fc_backbone1 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone2 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone3 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
        else:
            self.fc_backbone1 = nn.Identity()
            self.fc_backbone2 = nn.Identity()
            self.fc_backbone3 = nn.Identity()
            
        self.positional_embedding = positional_embedding
        self.cls_token = cls_token
        
        if self.cls_token:
            self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            # if self.backbone3 is not None:
            #     self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            num_patches = num_patches + 1
        if self.positional_embedding:
            self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        else:
            print('Do not use Pos Embedding!!')
            # if self.backbone3 is not None:
            #     self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        self.dropout = nn.Dropout(emb_dropout)
        # Cross Attention
        self.TF_layers = Encoder_blocks_three_inputs(emb_size=emb_size, shared_attention=shared_attention,
                    depth=depth,
                    heads= heads,dropout=dropout,last_gelu=last_gelu)

        self.avgpool = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(emb_size * 3, class_num) # emb_size * 4 ? 



    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        if y is None:
            y = x

        out1 = self.backbone1(x) # [batch,2048,7,7]
        out2 = self.backbone2(y) # [batch,2048,7,7]
        out3 = self.backbone3(z)

        batch_out1, emb_size_out1, height_out1, width_out1 = out1.size()
        batch_out2, emb_size_out2, height_out2, width_out2 = out2.size()
        batch_out3, emb_size_out3, height_out3, width_out3 = out3.size()

        out1 = out1.reshape(batch_out1,emb_size_out1,width_out1*height_out1)
        out2 = out2.reshape(batch_out2,emb_size_out2,width_out2*height_out2)
        out3 = out3.reshape(batch_out3,emb_size_out3,width_out3*height_out3)

        # [Batch, Channel, Patch] -> [Batch, Patch, Channel]
        out1 = out1.permute(0,2,1)
        out2 = out2.permute(0,2,1)
        out3 = out3.permute(0,2,1)

        out1 = self.fc_backbone1(out1) # [batch, 196, emb_size]
        out2 = self.fc_backbone2(out2) # [batch, 196, emb_size]
        out3 = self.fc_backbone3(out3)

        if self.cls_token:
            cls_token_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = batch_out1)
            cls_token_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = batch_out2)
            cls_token_3 = repeat(self.cls_token_3, '1 1 d -> b 1 d', b = batch_out3)

            out1 = torch.cat((cls_token_1,out1),dim=1)
            out2 = torch.cat((cls_token_2,out2),dim=1)
            out3 = torch.cat((cls_token_3,out3),dim=1)

        # Adding Positional Embedding
        if self.positional_embedding:
            # print(f'out1 shape = {out1.shape} // self.pos_embedding_1 shape = {self.pos_embedding_1.shape}')
            # exit(1)
            out1 = out1 + self.pos_embedding_1
            out2 = out2 + self.pos_embedding_2
            out3 = out3 + self.pos_embedding_3
        
        # dropout
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)
        out3 = self.dropout(out3)
        
        
        tf_output1, tf_output2, tf_output3 = self.TF_layers(out1,out2,out3)
        
        tf_output1 = tf_output1.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        tf_output2 = tf_output2.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        tf_output3 = tf_output3.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        
        if self.cls_token:
            tf_output1 = tf_output1[:,:,0]
            tf_output2 = tf_output2[:,:,0]
            tf_output3 = tf_output3[:,:,0]
        else:
            tf_output1 = self.avgpool(tf_output1).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            tf_output2 = self.avgpool(tf_output2).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            tf_output3 = self.avgpool(tf_output3).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
        
        combined = torch.cat((tf_output1, tf_output2,tf_output3), dim = -1) # [batch, emb_size*2]
        fc_out = self.fc(combined)
                
        return fc_out
    
class MHCA_Attention(nn.Module):
    def __init__(self, backbone1: nn.Module, backbone2: nn.Module,backbone3: nn.Module, backbone_freezing: bool=True,
                 shared_attention:bool=False,
                    backbone_output_dim: int=2048,emb_size: int=768, num_patches: int=196,depth: int=1,
                    positional_embedding: bool=True,cls_token: bool=False,last_gelu=False,
                    heads: int = 8,dropout:float=0.,emb_dropout:float=0., class_num=2):
# backbone3: nn.Module=None,
        super(MHCA_Attention, self).__init__()
        self.heads = heads
        # ResNet-50 (input :448)
        self.backbone1 = backbone1 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone2 = backbone2 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone3 = backbone3
        # if backbone3 is not None:
        #     self.backbone3 = backbone3

        # Freezing
        if backbone_freezing:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
            for param in self.backbone3.parameters():
                param.requires_grad = False
            # if backbone3 is not None:
            #     for param in self.backbone3.parameters():
            #         param.requires_grad = False
        if backbone_output_dim != emb_size: # emb_size = MHSA 
            self.fc_backbone1 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone2 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone3 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
        else:
            self.fc_backbone1 = nn.Identity()
            self.fc_backbone2 = nn.Identity()
            self.fc_backbone3 = nn.Identity()
            
        self.positional_embedding = positional_embedding
        self.cls_token = cls_token
        print(f'self.positional_embedding = {self.positional_embedding}')
        if self.cls_token:
            self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            # if self.backbone3 is not None:
            #     self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            num_patches = num_patches + 1
        if self.positional_embedding:
            self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        else:
            print('Do not use Pos Embedding!!')
            # if self.backbone3 is not None:
            #     self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        self.dropout = nn.Dropout(emb_dropout)
        # Cross Attention
        print(f'shared_attention = {shared_attention}')
        self.TF_layers = Encoder_blocks_three_inputs_MHCA(emb_size=emb_size, shared_attention=shared_attention,
                    depth=depth,
                    heads= heads,dropout=dropout,last_gelu=last_gelu)

        self.avgpool = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(emb_size * 3, class_num) # emb_size * 4 ? 



    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        if y is None:
            y = x

        out1 = self.backbone1(x) # [batch,2048,7,7]
        out2 = self.backbone2(y) # [batch,2048,7,7]
        out3 = self.backbone3(z)

        batch_out1, emb_size_out1, height_out1, width_out1 = out1.size()
        batch_out2, emb_size_out2, height_out2, width_out2 = out2.size()
        batch_out3, emb_size_out3, height_out3, width_out3 = out3.size()

        out1 = out1.reshape(batch_out1,emb_size_out1,width_out1*height_out1)
        out2 = out2.reshape(batch_out2,emb_size_out2,width_out2*height_out2)
        out3 = out3.reshape(batch_out3,emb_size_out3,width_out3*height_out3)

        # [Batch, Channel, Patch] -> [Batch, Patch, Channel]
        out1 = out1.permute(0,2,1)
        out2 = out2.permute(0,2,1)
        out3 = out3.permute(0,2,1)

        out1 = self.fc_backbone1(out1) # [batch, 196, emb_size]
        out2 = self.fc_backbone2(out2) # [batch, 196, emb_size]
        out3 = self.fc_backbone3(out3)

        if self.cls_token:
            cls_token_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = batch_out1)
            cls_token_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = batch_out2)
            cls_token_3 = repeat(self.cls_token_3, '1 1 d -> b 1 d', b = batch_out3)

            out1 = torch.cat((cls_token_1,out1),dim=1)
            out2 = torch.cat((cls_token_2,out2),dim=1)
            out3 = torch.cat((cls_token_3,out3),dim=1)

        # Adding Positional Embedding
        if self.positional_embedding:
            # print(f'out1 shape = {out1.shape} // self.pos_embedding_1 shape = {self.pos_embedding_1.shape}')
            # exit(1)
            out1 = out1 + self.pos_embedding_1
            out2 = out2 + self.pos_embedding_2
            out3 = out3 + self.pos_embedding_3
        
        # dropout
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)
        out3 = self.dropout(out3)
        
        
        tf_output1, tf_output2, tf_output3 = self.TF_layers(out1,out2,out3)
        
        tf_output1 = tf_output1.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        tf_output2 = tf_output2.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        tf_output3 = tf_output3.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        
        if self.cls_token:
            tf_output1 = tf_output1[:,:,0]
            tf_output2 = tf_output2[:,:,0]
            tf_output3 = tf_output3[:,:,0]
        else:
            tf_output1 = self.avgpool(tf_output1).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            tf_output2 = self.avgpool(tf_output2).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            tf_output3 = self.avgpool(tf_output3).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
        
        combined = torch.cat((tf_output1, tf_output2,tf_output3), dim = -1) # [batch, emb_size*2]
        fc_out = self.fc(combined)
                
        return fc_out
    
class MHDA_Attention(nn.Module):
    def __init__(self, backbone1: nn.Module, backbone2: nn.Module,backbone3: nn.Module, backbone_freezing: bool=True,
                 shared_attention:bool=False,
                    backbone_output_dim: int=2048,emb_size: int=768, num_patches: int=196,depth: int=1,
                    positional_embedding: bool=True,cls_token: bool=False,last_gelu=False,
                    heads: int = 8,dropout:float=0.,emb_dropout:float=0., class_num=2,last_mlps=False):
        super(MHDA_Attention, self).__init__()
        self.heads = heads
        # ResNet-50 (input :448)
        self.backbone1 = backbone1 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone2 = backbone2 # [B, 2048, 14, 14] Batch, channel, height, width
        self.backbone3 = backbone3
        # if backbone3 is not None:
        #     self.backbone3 = backbone3

        # Freezing
        if backbone_freezing:
            for param in self.backbone1.parameters():
                param.requires_grad = False
            for param in self.backbone2.parameters():
                param.requires_grad = False
            for param in self.backbone3.parameters():
                param.requires_grad = False
            # if backbone3 is not None:
            #     for param in self.backbone3.parameters():
            #         param.requires_grad = False
        if backbone_output_dim != emb_size: # emb_size = MHSA 
            self.fc_backbone1 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone2 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
            self.fc_backbone3 = nn.Linear(backbone_output_dim, emb_size) # 2048 -> 768
        else:
            self.fc_backbone1 = nn.Identity()
            self.fc_backbone2 = nn.Identity()
            self.fc_backbone3 = nn.Identity()
            
        self.positional_embedding = positional_embedding
        self.cls_token = cls_token
        
        if self.cls_token:
            self.cls_token_1 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_2 = nn.Parameter(torch.randn(1, 1, emb_size))
            self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            # if self.backbone3 is not None:
            #     self.cls_token_3 = nn.Parameter(torch.randn(1, 1, emb_size))
            num_patches = num_patches + 1
        if self.positional_embedding:
            self.pos_embedding_1 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, num_patches, emb_size))
            self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        else:
            print('Do not use Pos Embedding!!')
            # if self.backbone3 is not None:
            #     self.pos_embedding_3 = nn.Parameter(torch.randn(1, num_patches, emb_size))
        self.dropout = nn.Dropout(emb_dropout)

        # Cross Attention
        self.MHSA_layers = Encoder_blocks_three_inputs(emb_size=emb_size, shared_attention=shared_attention,
                    depth=depth,
                    heads= heads,dropout=dropout,last_gelu=last_gelu)
        self.MHCA_layers = Encoder_blocks_three_inputs_MHCA(emb_size=emb_size, shared_attention=shared_attention,
                    depth=depth,
                    heads= heads,dropout=dropout,last_gelu=last_gelu)

        self.avgpool = nn.AdaptiveAvgPool1d((1))
        if last_mlps:
            print('Using MLP!!')
            self.fc = nn.Sequential(nn.Linear(emb_size*6,emb_size),nn.ReLU(),nn.Linear(emb_size,class_num))
        else:
            print('Without MLP!!')
            self.fc = nn.Linear(emb_size * 6, class_num) # emb_size * 4 ? 



    def forward(self, x: Tensor, y: Tensor,z :Tensor) -> Tensor:
        out1 = self.backbone1(x) # [batch,2048,7,7]
        out2 = self.backbone2(y) # [batch,2048,7,7]
        out3 = self.backbone3(z)

        batch_out1, emb_size_out1, height_out1, width_out1 = out1.size()
        batch_out2, emb_size_out2, height_out2, width_out2 = out2.size()
        batch_out3, emb_size_out3, height_out3, width_out3 = out3.size()

        out1 = out1.reshape(batch_out1,emb_size_out1,width_out1*height_out1)
        out2 = out2.reshape(batch_out2,emb_size_out2,width_out2*height_out2)
        out3 = out3.reshape(batch_out3,emb_size_out3,width_out3*height_out3)

        # [Batch, Channel, Patch] -> [Batch, Patch, Channel]
        out1 = out1.permute(0,2,1)
        out2 = out2.permute(0,2,1)
        out3 = out3.permute(0,2,1)

        out1 = self.fc_backbone1(out1) # [batch, 196, emb_size]
        out2 = self.fc_backbone2(out2) # [batch, 196, emb_size]
        out3 = self.fc_backbone3(out3)

        if self.cls_token:
            cls_token_1 = repeat(self.cls_token_1, '1 1 d -> b 1 d', b = batch_out1)
            cls_token_2 = repeat(self.cls_token_2, '1 1 d -> b 1 d', b = batch_out2)
            cls_token_3 = repeat(self.cls_token_3, '1 1 d -> b 1 d', b = batch_out3)

            out1 = torch.cat((cls_token_1,out1),dim=1)
            out2 = torch.cat((cls_token_2,out2),dim=1)
            out3 = torch.cat((cls_token_3,out3),dim=1)

        # Adding Positional Embedding
        if self.positional_embedding:
            # print(f'out1 shape = {out1.shape} // self.pos_embedding_1 shape = {self.pos_embedding_1.shape}')
            # exit(1)
            out1 = out1 + self.pos_embedding_1
            out2 = out2 + self.pos_embedding_2
            out3 = out3 + self.pos_embedding_3
        
        # dropout
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)
        out3 = self.dropout(out3)
        
        
        MHSA_output1, MHSA_output2, MHSA_output3 = self.MHSA_layers(out1,out2,out3)
        MHCA_output1, MHCA_output2, MHCA_output3 = self.MHCA_layers(out1,out2,out3)
        
        MHSA_output1 = MHSA_output1.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        MHSA_output2 = MHSA_output2.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        MHSA_output3 = MHSA_output3.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]

        MHCA_output1 = MHCA_output1.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        MHCA_output2 = MHCA_output2.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        MHCA_output3 = MHCA_output3.permute(0,2,1) # [Batch, 196, emb_size] -> [Batch, emb_size, 196]
        
        if self.cls_token:
            MHSA_output1 = MHSA_output1[:,:,0]
            MHSA_output2 = MHSA_output2[:,:,0]
            MHSA_output3 = MHSA_output3[:,:,0]
            MHCA_output1 = MHCA_output1[:,:,0]
            MHCA_output2 = MHCA_output2[:,:,0]
            MHCA_output3 = MHCA_output3[:,:,0]
        else:
            MHSA_output1 = self.avgpool(MHSA_output1).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            MHSA_output2 = self.avgpool(MHSA_output2).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            MHSA_output3 = self.avgpool(MHSA_output3).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            MHCA_output1 = self.avgpool(MHCA_output1).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            MHCA_output2 = self.avgpool(MHCA_output2).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
            MHCA_output3 = self.avgpool(MHCA_output3).squeeze(2) # [Batch, emb_size, 196] -> [Batch, emb_size, 1] -> [Batch, emb_size]
        

        
        
        
        combined = torch.cat((MHSA_output1, MHSA_output2,MHSA_output3,MHCA_output1,MHCA_output2,MHCA_output3), dim = -1) # [batch, emb_size*2]
        fc_out = self.fc(combined)
                
        return fc_out