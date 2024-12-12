import torch
import botorch
import gpytorch
import tqdm.auto as tqdm
import botorch.acquisition.analytic
from botorch.utils.transforms import normalize

class GPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def fit_gp_model(train_x, train_y, epochs):
    noise = 1e-4

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood)
    model.likelihood.noise = noise

    # Train the hyperparameter
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr = 0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    model.train()
    likelihood.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

    return model, likelihood

   