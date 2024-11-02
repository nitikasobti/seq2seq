# # seq2seq_model.py
# import torch
# import torch.nn as nn

# class Encoder(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
#         super(Encoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

#     def forward(self, x):
#         embedding = self.embedding(x)
#         outputs, (hidden, cell) = self.lstm(embedding)
#         return hidden, cell

# class Decoder(nn.Module):
#     def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
#         super(Decoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x, hidden, cell):
#         embedding = self.embedding(x.unsqueeze(1))
#         outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
#         predictions = self.fc(outputs.squeeze(1))
#         return predictions, hidden, cell

# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(Seq2Seq, self).__init__()
#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, source, target, teacher_forcing_ratio=0.5):
#         batch_size = target.size(0)
#         target_len = target.size(1)
#         vocab_size = self.decoder.fc.out_features

#         outputs = torch.zeros(batch_size, target_len, vocab_size).to(target.device)
#         hidden, cell = self.encoder(source)

#         x = target[:, 0]
#         for t in range(1, target_len):
#             output, hidden, cell = self.decoder(x, hidden, cell)
#             outputs[:, t] = output
#             best_guess = output.argmax(1)
#             x = target[:, t] if torch.rand(1).item() < teacher_forcing_ratio else best_guess

#         return outputs
# seq2seq_model.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)  # [batch_size] -> [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # output: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sentence
        hidden, cell = self.encoder(src)
        
        # First input to the decoder is the <sos> tokens
        input = trg[:,0]
        
        for t in range(1, trg_len):
            # Insert input token embedding, previous hidden and previous cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            
            # Decide if we are going to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # If teacher forcing, use actual next token as next input; else, use predicted token
            input = trg[:, t] if teacher_force else top1
        
        return outputs
