import moments
import normflows as nf
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# Set up model
base = nf.distributions.base.DiagGaussian(2)


# Define list of flows
num_layers = 32
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 64 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([64, 64, 64, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
    
# Construct flow model
model = nf.NormalizingFlow(base, flows)

enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

# Log loss
loss_hist = np.array([])



for it in tqdm(range(100)):
    data = np.asarray([moments.Spectrum(moments.LinearSystem_1D.steady_state_1D(127, gamma=-10)) for i in range(0,20)])
    x = torch.poisson(torch.tensor(data)).to(device)
    loss = model.forward_kld(x)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    if (it + 1) % 10 == 0:
        model.eval()
        z, log_prob = model.sample(1)
        log_prob += z
        prob = torch.exp(log_prob.to('cpu'))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.plot(prob)
        plt.savefig('prob_{}.png'.format(it))
# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()