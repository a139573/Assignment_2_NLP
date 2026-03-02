import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, n_classes=20, dropout=0.3, bidirectional=True): # <-- Added parameter
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional) # <-- Use parameter
        
        # Calculate correct input size for the linear layer
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, n_classes)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        # Global Max Pooling for Human Values keywords
        pooled, _ = torch.max(lstm_out, dim=1)
        return self.fc(self.dropout(pooled))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)