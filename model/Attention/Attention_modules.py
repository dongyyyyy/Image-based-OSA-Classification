from . import *


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




class Encoder_block(nn.Module):
    def __init__(self, emb_size: int=768, cross_attention: bool=False,shared_attention=False,
                    inversed_cross_attention: bool=False, 
                    heads: int = 8,dropout: float = 0.):
        super(Encoder_block, self).__init__()
        self.shared_attention = shared_attention
        self.cross_attention = cross_attention
        self.inversed_cross_attention = inversed_cross_attention
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        # Cross Attention
        if self.shared_attention:
            if self.cross_attention:
                self.mhsa_1 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            else:
                self.mhsa_1 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        else:
            if self.cross_attention:
                self.mhsa_1 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
                self.mhsa_2 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            else:
                self.mhsa_1 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
                self.mhsa_2 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN2 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        #Feed-Forward Network
        
    def forward(self, x:Tensor, y:Tensor):
        if self.cross_attention: # Cross-Attention 
            #query, (key,value)
            if self.inversed_cross_attention:
                if self.shared_attention:
                    # print(f'using shared!!')
                    x = self.mhsa_1(x,y) + x 
                    y = self.mhsa_1(y,x) + y
                else:
                    # print(f'using none-shared!!')
                    x = self.mhsa_1(x,y) + x 
                    y = self.mhsa_2(y,x) + y
            else:
                if self.shared_attention:
                    # print(f'using shared!!')
                    x = self.mhsa_1(y,x) + x 
                    y = self.mhsa_1(x,y) + y
                else:
                    # print(f'using none-shared!!')
                    x = self.mhsa_1(y,x) + x 
                    y = self.mhsa_2(x,y) + y
        else: # Self-Attention
            if self.shared_attention:
                # print(f'using shared!!')
                x = self.mhsa_1(x) + x 
                y = self.mhsa_1(y) + y
            else:
                # print(f'using none-shared!!')
                x = self.mhsa_1(x) + x 
                y = self.mhsa_2(y) + y
        if self.shared_attention:
            # print(f'using shared!!')
            x = self.FFN1(x) + x
            y = self.FFN1(y) + y    
        else:
            # print(f'using none-shared!!')
            x = self.FFN1(x) + x
            y = self.FFN2(y) + y
        
        return x,y
    

class Encoder_blocks(nn.Module):
    def __init__(self, emb_size: int=768, cross_attention: bool=False,shared_attention:bool=False,
                    inversed_cross_attention: bool=False, depth:int=1,
                    heads: int = 8,dropout:float=0.,last_gelu: bool =False):
        super(Encoder_blocks, self).__init__()
        self.TF_layers = nn.ModuleList([])
        self.last_gelu=last_gelu
        if self.last_gelu:
            self.gelu = nn.GELU()
        for _ in range(depth):
            self.TF_layers.append(
                Encoder_block(emb_size=emb_size, cross_attention=cross_attention,shared_attention=shared_attention,
                    inversed_cross_attention=inversed_cross_attention, 
                    heads=heads,dropout=dropout)        
            )
    def forward(self,x,y):
        for layer in self.TF_layers:
            x, y = layer(x,y)
            if self.last_gelu:
                x = self.gelu(x)
                y = self.gelu(y)
        return x,y


class Encoder_block_three_inputs(nn.Module):
    def __init__(self, emb_size: int=768, cross_attention: bool=False,shared_attention=False,
                    inversed_cross_attention: bool=False, 
                    heads: int = 8,dropout: float = 0.):
        super(Encoder_block_three_inputs, self).__init__()
        self.shared_attention = shared_attention
        self.cross_attention = cross_attention
        self.inversed_cross_attention = inversed_cross_attention
        self.norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        # Cross Attention
        if self.shared_attention:
            if self.cross_attention:
                self.mhsa_1 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            else:
                self.mhsa_1 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        else:
            if self.cross_attention:
                self.mhsa_1 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
                self.mhsa_2 = MHSA_CA(dim = emb_size, heads = heads,dropout=dropout)
            else:
                self.mhsa_1 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
                self.mhsa_2 = MHSA(dim = emb_size, heads = heads,dropout=dropout)
            self.FFN1 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
            self.FFN2 = Feed_forward_networks(input_dim=emb_size,hidden_dim=emb_size*4,output_dim=emb_size,dropout=dropout)
        #Feed-Forward Network
        
    def forward(self, x:Tensor, y:Tensor):
        if self.cross_attention: # Cross-Attention 
            #query, (key,value)
            if self.inversed_cross_attention:
                if self.shared_attention:
                    # print(f'using shared!!')
                    x = self.mhsa_1(x,y) + x 
                    y = self.mhsa_1(y,x) + y
                else:
                    # print(f'using none-shared!!')
                    x = self.mhsa_1(x,y) + x 
                    y = self.mhsa_2(y,x) + y
            else:
                if self.shared_attention:
                    # print(f'using shared!!')
                    x = self.mhsa_1(y,x) + x 
                    y = self.mhsa_1(x,y) + y
                else:
                    # print(f'using none-shared!!')
                    x = self.mhsa_1(y,x) + x 
                    y = self.mhsa_2(x,y) + y
        else: # Self-AttentionCA
            if self.shared_attention:
                # print(f'using shared!!')
                x = self.mhsa_1(x) + x 
                y = self.mhsa_1(y) + y
            else:
                # print(f'using none-shared!!')
                x = self.mhsa_1(x) + x 
                y = self.mhsa_2(y) + y
        if self.shared_attention:
            # print(f'using shared!!')
            x = self.FFN1(x) + x
            y = self.FFN1(y) + y    
        else:
            # print(f'using none-shared!!')
            x = self.FFN1(x) + x
            y = self.FFN2(y) + y
        
        return x,y
    
