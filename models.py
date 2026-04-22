import torch
import torch.nn as nn
import math
import numpy as np

class LSTMModel(nn.Module):
    """LSTM model for NILM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        if self.bidirectional:
            # Concatenate the last outputs from both directions
            last_out = lstm_out[:, -1, :]
        else:
            # Just take the last output
            last_out = lstm_out[:, -1, :]
        
        # Linear layer
        output = self.fc(last_out)
        
        return output

class GRUModel(nn.Module):
    """GRU model for NILM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=False):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru(x)
        
        # Take the output from the last time step
        last_out = gru_out[:, -1, :]
        
        # Linear layer
        output = self.fc(last_out)
        
        return output

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size-1) * dilation // 2,
            dilation=dilation
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCNModel(nn.Module):
    """Temporal Convolutional Network model for NILM"""
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )
        
        self.tcn_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Permute to (batch_size, input_size, sequence_length) for 1D convolution
        x = x.permute(0, 2, 1)
        
        # TCN layers
        x = self.tcn_layers(x)
        
        # Global average pooling
        x = torch.mean(x, dim=2)
        
        # Linear layer
        x = self.fc(x)
        
        return x

class LiquidTimeLayer(nn.Module):
    """Liquid Time-Constant Neural Network Layer"""
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(LiquidTimeLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Time constants (initialized to 1.0)
        self.tau = nn.Parameter(torch.ones(hidden_size))
        
        # Recurrent weights
        self.rec_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        
        # Activation function
        self.tanh = nn.Tanh()
    
    def forward(self, x, hidden=None):
        """
        Forward pass with Euler integration
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Hidden state tensor of shape (batch_size, hidden_size)
            
        Returns:
            New hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Input projection
        input_proj = self.input_proj(x)
        
        # Recurrent projection
        rec_proj = torch.matmul(hidden, self.rec_weights)
        
        # ODE integration (Euler method)
        # dh/dt = -h/tau + f(Wx + Uh)
        tau = torch.nn.functional.softplus(self.tau).unsqueeze(0).clamp(min=1e-3)
        f_t = self.tanh(input_proj + rec_proj)
        dh = (-hidden / tau + f_t) * self.dt
        
        # Update hidden state
        new_hidden = hidden + dh
        
        return new_hidden

class LiquidNetworkModel(nn.Module):
    """Liquid Neural Network for NILM"""
    def __init__(self, input_size, hidden_size, output_size, dt=0.1):
        super(LiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Liquid time layer
        self.liquid_layer = LiquidTimeLayer(input_size, hidden_size, dt)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden state
        hidden = None
        
        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]
            hidden = self.liquid_layer(x_t, hidden)
        
        # Use the final hidden state for prediction
        output = self.fc(hidden)
        
        return output

# More sophisticated Liquid Neural Network implementation
class AdvancedLiquidTimeLayer(nn.Module):
    """Advanced Liquid Time-Constant Neural Network Layer with adaptive time constants"""
    def __init__(self, input_size, hidden_size, dt=0.1):
        super(AdvancedLiquidTimeLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Base time constants
        self.tau_base = nn.Parameter(torch.ones(hidden_size))
        
        # Adaptive time constants modulation
        self.tau_mod = nn.Linear(input_size, hidden_size)
        
        # Recurrent weights
        self.rec_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        
        # Gate for controlling information flow
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hidden=None):
        """
        Forward pass with advanced Euler integration
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Hidden state tensor of shape (batch_size, hidden_size)
            
        Returns:
            New hidden state
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Input projection
        input_proj = self.input_proj(x)
        
        # Recurrent projection
        rec_proj = torch.matmul(hidden, self.rec_weights)
        
        # Adaptive time constants (softplus keeps tau_base positive; clamp prevents /0)
        tau_mod = self.sigmoid(self.tau_mod(x))
        tau = torch.nn.functional.softplus(self.tau_base).unsqueeze(0) * tau_mod
        tau = tau.clamp(min=1e-3)
        
        # Input-dependent gate
        combined = torch.cat([x, hidden], dim=1)
        gate = self.sigmoid(self.gate(combined))
        
        # ODE integration with gating
        f_t = self.tanh(input_proj + rec_proj)
        dh = ((-hidden / tau) + gate * f_t) * self.dt
        
        # Update hidden state
        new_hidden = hidden + dh
        
        return new_hidden

class AdvancedLiquidNetworkModel(nn.Module):
    """Advanced Liquid Neural Network for NILM"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dt=0.1):
        super(AdvancedLiquidNetworkModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multiple liquid time layers
        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt
            ) for i in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor
        """
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_length, _ = x.size()
        
        # Initialize hidden states for each layer
        hidden_states = [None] * self.num_layers
        
        # Process each time step
        for t in range(seq_length):
            x_t = x[:, t, :]
            
            # Process through each layer
            for i in range(self.num_layers):
                if i == 0:
                    hidden_states[i] = self.liquid_layers[i](x_t, hidden_states[i])
                else:
                    hidden_states[i] = self.liquid_layers[i](hidden_states[i-1], hidden_states[i])
        
        # Use the final hidden state from the last layer for prediction
        output = self.fc(hidden_states[-1])
        
        return output

class ResidualBlock(nn.Module):
    """Residual block for ResNet architecture"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=5, 
            stride=stride, 
            padding=2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size=5, 
            padding=2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        # If dimensions change, we need to adjust the shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        # Store the input for the skip connection
        residual = x
        
        # First conv block
        out = self.relu(self.bn1(self.conv1(x)))
        
        # Second conv block (no activation yet)
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(residual)
        
        # Apply ReLU activation after addition
        out = self.relu(out)
        
        return out

class ResNetModel(nn.Module):
    """ResNet model for NILM"""
    def __init__(self, input_size, output_size, layers=[2, 2, 2], base_width=32):
        super(ResNetModel, self).__init__()
        
        self.in_channels = base_width
        
        # Initial convolutional layer
        self.conv1 = nn.Conv1d(input_size, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(base_width, layers[0])
        self.layer2 = self._make_layer(base_width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width*4, layers[2], stride=2)
        
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width*4, output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride=1):
        layers = []
        
        # First block may have stride > 1 to reduce spatial dimensions
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks (with stride=1)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Permute for 1D convolution: (batch_size, input_size, sequence_length)
        x = x.permute(0, 2, 1)
        
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class SimpleMultiHeadAttention(nn.Module):
    """Simple multi-head attention mechanism for lightweight transformer"""
    def __init__(self, embed_dim, num_heads=4):
        super(SimpleMultiHeadAttention, self).__init__()
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for query, key, value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _ = x.size()
        
        # Linear projections for Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for multi-head attention:
        # (batch_size, seq_length, embed_dim) -> (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions:
        # (batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Final output projection
        output = self.out_proj(context)
        
        return output

class TransformerEncoderLayer(nn.Module):
    """Simplified Transformer encoder layer"""
    def __init__(self, embed_dim, num_heads=4, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = SimpleMultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Activation function
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Self-attention block with residual connection and layer norm
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = residual + self.dropout(x)
        
        # Feed-forward block with residual connection and layer norm
        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = residual + self.dropout(x)
        
        return x

class SimplePositionalEncoding(nn.Module):
    """Simple positional encoding for transformer models"""
    def __init__(self, embed_dim, max_len=5000):
        super(SimplePositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Sine for even indices, cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_dim)
        # Add positional encoding to the input
        return x + self.pe[:, :x.size(1), :]

class SimpleTransformerModel(nn.Module):
    """Simplified Transformer model for NILM"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=4, dropout=0.1):
        super(SimpleTransformerModel, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding
        self.pos_encoding = SimplePositionalEncoding(hidden_size)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size, 
                num_heads=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global average pooling over the sequence dimension
        x = torch.mean(x, dim=1)
        
        # Output layer
        x = self.output_layer(x)
        
        return x