import torch
from torch import nn


class LuongAttention(nn.Module):
    """Implements General Luong Attention
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.A = nn.Linear(hidden_size, hidden_size)
        self.C = nn.Linear(hidden_size*2, hidden_size)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

    def forward(self, hidden_state, encoder_outputs):
        # hidden states of shape (N, 1, hidden_size)
        # encoder outputs of shape (N, S, T)

        hidden_state = self.A(hidden_state)  # (N, 1, hidden_size)

        # (N, S, T) @ (N, T, 1) -> (N, S, 1)

        scores = torch.bmm(encoder_outputs, self.A(
            hidden_state).transpose(-1, -2))
        scores = self.softmax(scores)

        # (N, 1, S) @ (N, S, T) -> (N, 1, T)
        context = torch.bmm(scores.transpose(-1, -2), encoder_outputs)

        concat = torch.cat((hidden_state.squeeze(1), context.squeeze(1)), 1)
        concat = self.tanh(self.C(concat))  # (N, T)

        return concat, scores