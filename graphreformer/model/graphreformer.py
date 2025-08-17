import torch
import torch.nn as nn
from reformer_pytorch import Reformer

class DFSGraphTransformer(nn.Module):
    def __init__(
        self,
        num_node_ids,
        num_node_labels,
        num_edge_labels,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        pad_token_id=0
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_field = d_model // 5  # Each field gets one slice
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Field embeddings (separate)
        self.embed_v1 = nn.Embedding(num_node_ids, self.d_field, padding_idx=pad_token_id)
        self.embed_v2 = nn.Embedding(num_node_ids, self.d_field, padding_idx=pad_token_id)
        self.embed_l1 = nn.Embedding(num_node_labels, self.d_field, padding_idx=pad_token_id)
        self.embed_e  = nn.Embedding(num_edge_labels, self.d_field, padding_idx=pad_token_id)
        self.embed_l2 = nn.Embedding(num_node_labels, self.d_field, padding_idx=pad_token_id)

        # Project concatenated fields to d_model
        self.input_projection = nn.Linear(5 * self.d_field, d_model)

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.out_v1 = nn.Linear(d_model, num_node_ids)
        self.out_v2 = nn.Linear(d_model, num_node_ids)
        self.out_l1 = nn.Linear(d_model, num_node_labels)
        self.out_e  = nn.Linear(d_model, num_edge_labels)
        self.out_l2 = nn.Linear(d_model, num_node_labels)

    def generate_causal_mask(self, L, device):
        return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)

    def rotary_positional_embedding(self, x, seq_len):
        """
        Applies rotary positional embeddings to the input tensor.
        
        Args:
            x: Input tensor of shape (B, L, d_model)
            seq_len: Sequence length L
            
        Returns:
            Tensor with rotary embeddings applied
        """
        B, L, _ = x.shape
        device = x.device
        
        # Create position indices [0, 1, 2, ..., L-1]
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        
        # Create dimension indices [0, 1, 2, ..., d_model//2 - 1]
        dim = torch.arange(self.head_dim // 2, dtype=torch.float, device=device).unsqueeze(0)
        
        # Calculate the angles for the rotary embeddings
        freq = 1.0 / (10000.0 ** (2 * dim / self.head_dim))
        angles = pos * freq
        
        # Compute cos and sin values
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        
        # Reshape x to separate heads
        x = x.view(B, L, self.nhead, self.head_dim)
        
        # Split into two parts along the head dimension
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        x_rot = torch.cat((-x2, x1), dim=-1)
        
        # Combine using cos and sin
        x = (x * cos.unsqueeze(1)) + (x_rot * sin.unsqueeze(1))
        
        # Reshape back to original shape
        return x.reshape(B, L, self.d_model)

    def forward(self, inputs):
        """
        inputs: dict of tensors, keys = ['v1', 'v2', 'l1', 'e', 'l2'], each (B, L)
        """
        B, L = inputs['v1'].shape
        device = inputs['v1'].device

        # Field embeddings
        emb_v1 = self.embed_v1(inputs['v1'])
        emb_v2 = self.embed_v2(inputs['v2'])
        emb_l1 = self.embed_l1(inputs['l1'])
        emb_e  = self.embed_e(inputs['e'])
        emb_l2 = self.embed_l2(inputs['l2'])

        # Concatenate and project
        emb_cat = torch.cat([emb_v1, emb_v2, emb_l1, emb_e, emb_l2], dim=-1)  # (B, L, 5*d_field)
        x = self.input_projection(emb_cat)  # (B, L, d_model)

        # Apply rotary positional embeddings
        x = self.rotary_positional_embedding(x, L)

        # Masking
        causal_mask = self.generate_causal_mask(L, device)
        pad_mask = inputs['v1'] == self.pad_token_id

        # Transformer forward
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        return {
            'v1': self.out_v1(x),
            'v2': self.out_v2(x),
            'l1': self.out_l1(x),
            'e':  self.out_e(x),
            'l2': self.out_l2(x),
        }
    

class DFSGraphReformer(nn.Module):
    def __init__(
        self,
        num_node_ids,
        num_node_labels,
        num_edge_labels,
        bucket_size=64,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        pad_token_id=0,
        max_seq_len=1024
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_field = d_model // 5
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.embed_v1 = nn.Embedding(num_node_ids, self.d_field, padding_idx=pad_token_id)
        self.embed_v2 = nn.Embedding(num_node_ids, self.d_field, padding_idx=pad_token_id)
        self.embed_l1 = nn.Embedding(num_node_labels, self.d_field, padding_idx=pad_token_id)
        self.embed_e  = nn.Embedding(num_edge_labels, self.d_field, padding_idx=pad_token_id)
        self.embed_l2 = nn.Embedding(num_node_labels, self.d_field, padding_idx=pad_token_id)

        self.input_projection = nn.Linear(5 * self.d_field, d_model)

        # Reformer setup
        self.reformer = Reformer(
            dim=d_model,
            depth=num_layers,
            heads=nhead,
            lsh_dropout=dropout,
            ff_chunks=8,
            causal=True,
            bucket_size=bucket_size,
            ff_mult=dim_feedforward // d_model,
            use_full_attn=True,  # fallback to full attn for small seqs
        )

        # Output heads
        self.out_v1 = nn.Linear(d_model, num_node_ids)
        self.out_v2 = nn.Linear(d_model, num_node_ids)
        self.out_l1 = nn.Linear(d_model, num_node_labels)
        self.out_e  = nn.Linear(d_model, num_edge_labels)
        self.out_l2 = nn.Linear(d_model, num_node_labels)

    def forward(self, inputs):
        """
        inputs: dict of tensors, keys = ['v1', 'v2', 'l1', 'e', 'l2'], each (B, L)
        """
        B, L = inputs['v1'].shape
        device = inputs['v1'].device

        # Embeddings
        emb_v1 = self.embed_v1(inputs['v1'])
        emb_v2 = self.embed_v2(inputs['v2'])
        emb_l1 = self.embed_l1(inputs['l1'])
        emb_e  = self.embed_e(inputs['e'])
        emb_l2 = self.embed_l2(inputs['l2'])

        # Concatenate and project
        emb_cat = torch.cat([emb_v1, emb_v2, emb_l1, emb_e, emb_l2], dim=-1)
        x = self.input_projection(emb_cat)

        # Padding mask
        pad_mask = inputs['v1'] == self.pad_token_id  # (B, L)

        # Reformer forward
        x = self.reformer(x, input_mask=~pad_mask)

        return {
            'v1': self.out_v1(x),
            'v2': self.out_v2(x),
            'l1': self.out_l1(x),
            'e':  self.out_e(x),
            'l2': self.out_l2(x),
        }