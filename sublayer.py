import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUGate(nn.Module):
    def __init__(self, d_model):
        super(GRUGate,self).__init__()

        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(d_model, d_model)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)

        self.init_bias()

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)

    def forward(self, x, y):
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))
        return (1.-z)*x + z*h_hat


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
        self.ln= nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_k, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_model)
            
    def forward(self, x, adj):
        residual = x
        x = self.ln(x)
        q = x
        k = x
        v = x

        d_k, n_head = self.d_k, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_k) # (n*b) x lv x dv

        adj = adj.unsqueeze(1).repeat(1, n_head, 1, 1).reshape(-1, len_q, len_q)
        output = self.attention(q, k, v, adj)
        output = output.view(n_head, sz_b, len_q, d_k)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
    
        output = F.relu(self.dropout(self.fc(output)))
        output = self.gate(residual,output)
        return output  


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, dhid, dropout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, dhid)
        self.w_2 = nn.Linear(dhid, d_in)
        self.ln = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.gate = GRUGate(d_in)
            
    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = F.relu(self.w_2(F.relu((self.w_1(x)))))
        return self.gate(residual, x)
        
        
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, adj):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = attn.masked_fill(adj == 0, -np.inf)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output