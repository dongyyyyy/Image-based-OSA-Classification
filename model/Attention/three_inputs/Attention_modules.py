from .. import *


def backbone_load_weight(model,model_weight_path):
    
    check = torch.load(model_weight_path,weights_only=True)
    state_dict = check['model_state_dict']
    
    model.load_state_dict(state_dict, strict = True)
    
    
    
    return model
            


class MHSA(nn.Module):
    def __init__(self, dim: int, heads: int = 8,dropout:float=0.):
        super(MHSA, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim = dim, num_heads = heads,dropout=dropout,batch_first=True) # batch_first = False
        
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)  
        attn_output, _ = self.attn(x, x, x) #
        return attn_output

class MHSA_CA(nn.Module):
    def __init__(self, dim: int, heads: int = 8,dropout:float=0.):
        super(MHSA_CA, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim = dim, num_heads = heads,dropout=dropout,batch_first=True) # batch_first = False
        
        
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.norm(x)  
        y = self.norm(y)  

        attn_output, _ = self.attn(x, y, y) #
        

        return attn_output


class Feed_forward_networks(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int,dropout:float = 0.):
        super(Feed_forward_networks, self).__init__()
        self.mlp_view = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self,input):
        return self.mlp_view(input)



class Encoder_block_three_inputs(nn.Module):
    def __init__(self, emb_size: int=768, shared_attention=False,
                    heads: int = 8,dropout: float = 0.):
        super(Encoder_block_three_inputs, self).__init__()
        self.shared_attention = shared_attention
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        # Cross Attention
        if self.shared_attention:
            self.mhsa_1 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        else:

            self.mhsa_1 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhsa_2 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhsa_3 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN2 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN3 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        #Feed-Forward Network
        
    def forward(self, x:Tensor, y:Tensor, z:Tensor):
        
        if self.shared_attention:
            # print(f'using shared!!')
            x = self.mhsa_1(x) + x 
            y = self.mhsa_1(y) + y
            z = self.mhsa_1(z) + z
        else:
            # print(f'using none-shared!!')
            x = self.mhsa_1(x) + x 
            y = self.mhsa_2(y) + y
            z = self.mhsa_3(z) + z
        if self.shared_attention:
            # print(f'using shared!!')
            x = self.FFN1(x) + x
            y = self.FFN1(y) + y  
            z = self.FFN1(z) + z    
        else:
            # print(f'using none-shared!!')
            x = self.FFN1(x) + x
            y = self.FFN2(y) + y
            z = self.FFN3(z) + z
        
        return x,y,z
    

class Encoder_blocks_three_inputs(nn.Module):
    def __init__(self, emb_size: int=768, shared_attention:bool=False,
                  depth:int=1,
                    heads: int = 8,dropout:float=0.,last_gelu: bool =False):
        super(Encoder_blocks_three_inputs, self).__init__()
        self.TF_layers = nn.ModuleList([])
        self.last_gelu=last_gelu
        if self.last_gelu:
            self.gelu = nn.GELU()
        for _ in range(depth):
            self.TF_layers.append(
                Encoder_block_three_inputs(emb_size=emb_size,shared_attention=shared_attention,
                    heads=heads,dropout=dropout)        
            )
    def forward(self,x,y,z):
        for layer in self.TF_layers:
            x, y, z = layer(x,y,z)
            if self.last_gelu:
                x = self.gelu(x)
                y = self.gelu(y)
                z = self.gelu(z)
        return x,y,z


class Encoder_block_three_inputs_MHCA(nn.Module):
    def __init__(self, emb_size: int=768, shared_attention=False,
                    heads: int = 8,dropout: float = 0.):
        super(Encoder_block_three_inputs_MHCA, self).__init__()
        self.shared_attention = shared_attention
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        # Cross Attention
        if self.shared_attention:
            self.mhca_1 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        else:
            self.mhca_12 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhca_13 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhca_21 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhca_23 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhca_31 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            self.mhca_32 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)

            self.FFN12 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN13 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN21 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN23 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN31 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN32 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        #Feed-Forward Network
        
    def forward(self, x:Tensor, y:Tensor, z:Tensor):
        
        if self.shared_attention:
            # print(f'using shared!!')
            x1 = self.mhca_1(y,x) + x
            x2 = self.mhca_1(z,x) + x 
            y1 = self.mhca_1(x,y) + y
            y2 = self.mhca_1(z,y) + y
            z1 = self.mhca_1(x,z) + z
            z2 = self.mhca_1(y,z) + z
        else:
            # print(f'using none-shared!!')
            x1 = self.mhca_12(y,x) + x
            x2 = self.mhca_13(z,x) + x 
            y1 = self.mhca_21(x,y) + y
            y2 = self.mhca_23(z,y) + y
            z1 = self.mhca_31(x,z) + z
            z2 = self.mhca_32(y,z) + z
        if self.shared_attention:
            # print(f'using shared!!')
            x1 = self.FFN1(x1) + x1
            x2 = self.FFN1(x2) + x2
            y1 = self.FFN1(y1) + y1  
            y2 = self.FFN1(y2) + y2  
            z1 = self.FFN1(z1) + z1
            z2 = self.FFN1(z2) + z2     
        else:
            # print(f'using none-shared!!')
            x1 = self.FFN12(x1) + x1
            x2 = self.FFN13(x2) + x2
            y1 = self.FFN21(y1) + y1  
            y2 = self.FFN23(y2) + y2  
            z1 = self.FFN31(z1) + z1
            z2 = self.FFN32(z2) + z2

        x = x1 + x2
        y = y1 + y2
        z = z1 + z2
        
        return x,y,z

class Encoder_blocks_three_inputs_MHCA(nn.Module):
    def __init__(self, emb_size: int=768, shared_attention:bool=False,
                  depth:int=1,
                    heads: int = 8,dropout:float=0.,last_gelu: bool =False):
        super(Encoder_blocks_three_inputs_MHCA, self).__init__()
        self.TF_layers = nn.ModuleList([])
        self.last_gelu=last_gelu
        if self.last_gelu:
            self.gelu = nn.GELU()
        for _ in range(depth):
            self.TF_layers.append(
                Encoder_block_three_inputs_MHCA(emb_size=emb_size,shared_attention=shared_attention,
                    heads=heads,dropout=dropout)        
            )
    def forward(self,x,y,z):
        for layer in self.TF_layers:
            x, y, z = layer(x,y,z)
            if self.last_gelu:
                x = self.gelu(x)
                y = self.gelu(y)
                z = self.gelu(z)
        return x,y,z

