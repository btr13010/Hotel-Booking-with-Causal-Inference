import numpy as np
import torch
from torch import nn

np.random.seed(42)
g = torch.Generator().manual_seed(42)

# a neural net with the first 3 hidden layers then separate into 2 sub-networks with 2 hidden layers computing the potential outcome in treatment and control group
class Tarnet(nn.Module):
    def __init__(
        self,
        input_dim : int,
        hidden_units=20,
    ):
        super(Tarnet, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units,hidden_units),
            nn.ELU()
        )
        self.regressor1 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units//2),
            nn.ELU(),
            nn.Linear(hidden_units//2, hidden_units//2),
            nn.ELU(),
            nn.Linear(hidden_units//2, 1)
        )
        self.regressor2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units//2),
            nn.ELU(),
            nn.Linear(hidden_units//2, hidden_units//2),
            nn.ELU(),
            nn.Linear(hidden_units//2, 1)
        )
        
    def forward(self, x):
        x = self.linear_stack(x)
        out1 = self.regressor1(x)
        out2 = self.regressor2(x)
        concat = torch.cat((out1, out2),1)
        return concat

def loss(concat_true, concat_pred):
        """
        concat_true - 2 columns: outcome and treatment values
        concat_pred - 2 columns: outcome in treatment and control groups
        loss function: MSE - computed with the corresponding group (treatment or control)
        """
        y_true = concat_true[:,0] # true PO
        t_true = concat_true[:,1] # treatment value (0 or 1)

        y0_pred = concat_pred[:,0] # PO in control group
        y1_pred = concat_pred[:,1] # PO in treatment group

        # loss = t * (y1 - y_true)^2 + (1-t) * (y0 - y_true)^2
        loss = torch.sum((1-t_true) * torch.square(y0_pred - y_true) + t_true * torch.square(y1_pred - y_true))

        return loss

def estimate(
    treatment,
    outcome,
    confounders,
    batch_size = 32,
    n_iter = 2000,
    lr = 0.001,
):
    concat_true = torch.cat((outcome,treatment),1)
    input_dim = confounders.size(dim=1)

    model = Tarnet(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training
    for i in range(n_iter+1):
        # random batch
        idx = np.random.choice(len(concat_true), batch_size)
        concat_true_batch = concat_true[idx]
        w_batch = confounders[idx]

        # forward pass
        concat_pred = model(w_batch)
        loss_value = loss(concat_true_batch, concat_pred)

        # backward pass
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Iteration " + str(i) + " loss: " + str(loss_value.item()))

    concat_pred = model(confounders)
    y0_pred = concat_pred[:,0]
    y1_pred = concat_pred[:,1]
    cate_pred = y1_pred - y0_pred
    ate_pred = cate_pred.mean().item()

    return ate_pred