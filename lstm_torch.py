import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim

        # We need two Linears per gate: one for x_t and one for x_tm1.
        self.forget_gate_x, self.forget_gate_h = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.input_gate_x, self.input_gate_h  = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.cell_gate_x, self.cell_gate_h = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.output_gate_x, self.output_gate_h = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        
    def gate():
        return torch.sigmoid()
        
    def forward(self, x_t, state_tm1):
        # Unpack the previous state
        h_tm1, c_tm1 = state_tm1
        
        # Forget gate. (How much of the previous cell state to forget.)
        f_t = torch.sigmoid(self.forget_gate_x(x_t) + self.forget_gate_h(h_tm1))
        
        # Candiadate gate. (New values to add to the previous cell state.)
        g_t = torch.tanh(self.cell_gate_x(x_t) + self.cell_gate_h(h_tm1))
        
        # Input gate. (How much of new values to add.)
        i_t = torch.sigmoid(self.input_gate_x(x_t) + self.input_gate_h(h_tm1))
    
        # Update cell state.
        c_t = f_t * c_tm1 + i_t * g_t
        
        # Ouput gate. (Which parts of the cell state we will ouput to the hidden state.)
        o_t = torch.sigmoid(self.output_gate_x(x_t) + self.output_gate_h(h_tm1))
        
        # Update hidden state. (map cell state to -1,1 and choose how much of each.)
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.num_layers = num_layers
        
        # Multiple layers 
        self.layers = nn.ModuleList([LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
    
    def forward(self, x):
        n_batch, T, _ = x.size()
        
        # Hidden and cell states start with zeros.
        h = torch.zeros(n_batch, self.num_layers, self.hidden_dim)
        c = torch.zeros(n_batch, self.num_layers, self.hidden_dim)
        
        outputs = []
        
        # Loop over time steps.
        for t in range(T):
            x_t = x[:, t, :]
            
            # Normal forward.
            for i, layer in enumerate(self.layers):
   
                # Process through LSTM cell passing each its h_tm1, c_tm1 and update them.
                h[:, i], c[:, i] = layer(x_t, (h[:, i], c[:, i]))
                
                # Output of this layer becomes input to the next layer
                x_t = h[:, i]
            
            # Collect output from last layer
            outputs.append(x_t.unsqueeze(1))
        
        # Concat along sequence dimension
        return torch.cat(outputs, dim=1), (h, c)

if __name__ == "__main__":
    # https://karpathy.github.io/2015/05/21/rnn-effectiveness/
    # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    
    n_batch, T, input_size, hidden_size = 10, 252, 20, 40
    
    X = torch.randn(n_batch, T, input_size)
    
    lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
    output, (h, c) = lstm(X)
    
    homemade_lstm = LSTM(input_size, hidden_size, num_layers=2)
    output_homemade, (h_homemade, c_homemade) = homemade_lstm(X)
    
    
    print(f"{output.shape=}, {output_homemade.shape=}")
    print(f"{h.shape=}, {h_homemade.shape=}")
    print(f"{c.shape=}, {c_homemade.shape=}")