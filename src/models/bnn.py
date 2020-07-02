# credit: https://www.nitarshan.com/bayes-by-backprop/
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
import pandas as pd
from src.ml_helpers import duplicate, spread, get_unique_dir
from src.models.base import ProbModelBaseClass
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])

# SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
# SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

class Gaussian(object):
    def __init__(self, mu, rho, device):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
        self.size = self.rho.size()
        self.device = device
        self.sum_dim = [1,2] if len(self.size) == 2 else [1]

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self, S=1):
        epsilon = self.normal.sample([S, *self.size]).to(self.device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum(self.sum_dim)


class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)

    # replace w/ logsumexp
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        sum_dim = [1,2] if input.ndim == 3 else [1]
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum(sum_dim)


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, args):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho, args.device)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho, args.device)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_guide = 0

    def forward(self, input, sample=False, S=1):
        if self.training or sample:
            weight = self.weight.sample(S=S)
            bias = self.bias.sample(S=S)
        else:
            weight = self.weight.mu.unsqueeze(0)
            bias = self.bias.mu.unsqueeze(0)

        self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
        self.log_guide = self.weight.log_prob(weight) + self.bias.log_prob(bias)

        # vectorized
        return torch.matmul(input, weight.permute(0,2,1)) + bias.unsqueeze(1)


class BayesianNetwork(ProbModelBaseClass):
    def __init__(self, D, num_batches, args, num_classes=10):
        super(BayesianNetwork, self).__init__(D, args)
        self.NUM_BATCHES = num_batches # M in eq. 8 https://arxiv.org/pdf/1505.05424.pdf
        self.NUM_CLASSES = num_classes # M in eq. 8 https://arxiv.org/pdf/1505.05424.pdf

        self.l1 = BayesianLinear(D, 400, args)
        self.l2 = BayesianLinear(400, 400, args)
        self.l3 = BayesianLinear(400, 10, args)
        self.tensorboard_dir = Path(args.artifact_dir) / get_unique_dir()
        self.writer = SummaryWriter(self.tensorboard_dir)


    def set_internals(self, data, S):
        assert isinstance(data, (tuple, list)), "Data must be a tuple (X,y)"
        assert len(data) == 2, "Data must be a tuple (X,y)"
        self.x = data[0].to(self.args.device)
        self.y = data[1].to(self.args.device)
        self.z = self.sample_latent(S, sample=True)
        self.check_internals()

    def check_internals(self):
        """Verify internal state variables have been set.
         - False means not used,
         - None means error
        """
        assert self.x is not None, "self.x not set"
        assert self.y is not None, "self.y not set"
        assert self.z is not None, "self.z not set"

    def sample_latent(self, S, sample=True):
        # sample=False used for eval
        x = self.x.view(-1, self.D)
        x = F.relu(self.l1(x, sample=sample, S=S))
        x = F.relu(self.l2(x, sample=sample, S=S))
        return F.log_softmax(self.l3(x, sample=sample, S=S), dim=2)

    def log_prior(self):
        N = self.x.shape[0]
        log_prior = self.l1.log_prior + self.l2.log_prior + self.l3.log_prior
        return spread(log_prior, N) # to obey [NxS] output requirement

    def log_guide(self):
        N = self.x.shape[0]
        log_guide = self.l1.log_guide + self.l2.log_guide + self.l2.log_guide
        return spread(log_guide, N) # to obey [NxS] output requirement

    def log_likelihood(self):
        S = self.z.shape[0]
        # permute to obey [NxS] output requirement
        negative_log_likelihood = F.nll_loss(self.z.permute(1,2,0), duplicate(self.y, S, axis=1),  reduction='none')
        return -negative_log_likelihood

    def elbo(self):
        # this is the mini-batch weighted loss discussed in eq. 8 of the BBB paper.
        # Same trick works for tvo terms
        return (self.log_likelihood() + (self.log_prior() - self.log_guide())/self.NUM_BATCHES)

    # def elbo(self):
    #     # this is the mini-batch weighted loss discussed in eq. 8 of the BBB paper.
    #     # We multiply by N so torch.mean(elbo) in base.py is a sum to match eq. 8
    #     # Same trick works for tvo terms
    #     N = self.x.shape[0]
    #     return N*(self.log_likelihood() + (self.log_prior() - self.log_guide())/self.NUM_BATCHES)


    def get_prediction(self):
        self.check_internals()
        outputs = torch.zeros(self.args.test_S+1, self.args.test_batch_size, self.NUM_CLASSES).to(self.args.device)
        outputs[:self.args.test_S] = self.z
        outputs[self.args.test_S] = self.sample_latent(S=1, sample=False)
        output = outputs.mean(0)
        preds = preds = outputs.max(2, keepdim=True)[1]
        pred = output.max(1, keepdim=True)[1] # index of max log-probability

        sample_prediction = preds.eq(self.y.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
        ensemble_prediction = pred.eq(self.y.view_as(pred)).sum().item()

        return sample_prediction, ensemble_prediction

    def record_stats(self, loss=None, record_partition=False, epoch=None, batch_idx=None):
        if batch_idx == 0:
            self.write_weight_histograms(epoch)

        elbo = self.elbo().mean()
        log_prior = self.log_prior().mean()
        log_guide = self.log_guide().mean()
        negative_log_likelihood  = -self.log_likelihood().mean()

        global_step = epoch*self.NUM_BATCHES+batch_idx

        self.writer.add_scalar('logs/loss', loss, global_step)
        self.writer.add_scalar('logs/elbo', loss, global_step)
        self.writer.add_scalar('logs/complexity_cost', log_guide-log_prior, global_step)
        self.writer.add_scalar('logs/log_prior', log_prior, global_step)
        self.writer.add_scalar('logs/log_guide', log_guide, global_step)
        self.writer.add_scalar('logs/negative_log_likelihood', negative_log_likelihood, global_step)


    def write_weight_histograms(self, epoch):
        self.writer.add_histogram('histogram/w1_mu', self.l1.weight_mu, epoch)
        self.writer.add_histogram('histogram/w1_rho', self.l1.weight_rho, epoch)
        self.writer.add_histogram('histogram/w2_mu', self.l2.weight_mu, epoch)
        self.writer.add_histogram('histogram/w2_rho', self.l2.weight_rho, epoch)
        self.writer.add_histogram('histogram/w3_mu', self.l3.weight_mu, epoch)
        self.writer.add_histogram('histogram/w3_rho', self.l3.weight_rho, epoch)
        self.writer.add_histogram('histogram/b1_mu', self.l1.bias_mu, epoch)
        self.writer.add_histogram('histogram/b1_rho', self.l1.bias_rho, epoch)
        self.writer.add_histogram('histogram/b2_mu', self.l2.bias_mu, epoch)
        self.writer.add_histogram('histogram/b2_rho', self.l2.bias_rho, epoch)
        self.writer.add_histogram('histogram/b3_mu', self.l3.bias_mu, epoch)
        self.writer.add_histogram('histogram/b3_rho', self.l3.bias_rho, epoch)


    def evaluate_model_and_inference_network(self, data_loader, epoch=None):
        """
        overwrite evaluate_model_and_inference_network to add classification
        evaluation as well. this avoids having to add bnn-specific conditionals in main.py
        """
        self.eval()
        correct = 0
        corrects = np.zeros(self.args.test_S+1, dtype=int)
        TEST_SIZE = len(data_loader.dataset)

        log_p_total = 0
        kl_total = 0
        num_data = 0

        with torch.no_grad():
            data_loader = tqdm(data_loader) if self.args.verbose else data_loader
            for data in iter(data_loader):
                log_p, kl = self.get_log_p_and_kl(data, self.args.test_S)
                sample_prediction, ensemble_prediction = self.get_prediction()

                corrects += sample_prediction
                correct += ensemble_prediction

                log_p_total += torch.sum(log_p).item()
                kl_total += torch.sum(kl).item()

        df = pd.DataFrame(np.append(corrects, correct)) / TEST_SIZE

        df.index = [f"sample_{i}" for i in range(self.args.test_S)] + ["posterior"] + ["ensemble"]
        # df.to_csv(self.tensorboard_dir / f"classification_accuracy_epoch_{epoch}.csv")

        for name, row in df.iterrows():
            self.writer.add_scalar(f'accuracy/{name}', float(row), epoch)
            self.args._run.log_scalar(name, float(row), epoch)

        print(df)

        return log_p_total / TEST_SIZE, kl_total / TEST_SIZE


    def save_record(self):
        # here just to overwrite base class's save_record
        pass
