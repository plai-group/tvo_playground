from torch import nn
import abc
from src import ml_helpers as mlh
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import torch

from src.util import compute_tvo_loss, compute_wake_theta_loss, compute_wake_phi_loss, compute_vimco_loss, compute_tvo_reparam_loss, compute_iwae_loss
from src import assertions
from src import util


class ProbModelBaseClass(nn.Module):
    def __init__(self, D, args):
        """Base class for probabilistic model.
            - Uses internal state to avoid having to pass data around
            - self.set_internals(), self.log_guide(), self.log_prior(), self.sample_latent(),
              must be overwritten by subclass

        Args:
            D (int): [Size of observation dimension]
            S (int, optional): [Number of samples to be used in MC approx]. Defaults to 25.
        """
        super().__init__()

        # Dimensions
        self.D = D
        self.args = args

        if self.args.loss in assertions.REQUIRES_REPARAM:
            print("Reparam turned: ON")
            self.reparam = True
        else:
            print("Reparam turned: OFF")
            self.reparam = False

        # Internal state
        self.x = None  # Observations
        self.y = None  # Labels
        self.z = None  # Latent samples

        # for dual losses and optimizers, the convention is [theta, phi]
        self.losses = {
            'reinforce'  : [self.get_reinforce_loss],
            'elbo'       : [self.get_elbo_loss],
            'iwae'       : [self.get_iwae_loss],
            'tvo'        : [self.get_tvo_loss],
            'vimco'      : [self.get_vimco_loss],
            'tvo_reparam': [self.get_tvo_loss, self.get_tvo_reparam_loss],       # theta optimized using tvo,  phi using tvo_reparam
            'iwae_dreg'  : [self.get_iwae_loss, self.get_iwae_dreg_loss],        # theta optimized using IWAE, phi using IWAE_reparam
            'wake-sleep' : [self.get_wake_theta_loss, self.get_sleep_phi_loss],  # theta optimized using wake, phi using sleep
            'tvo-wake'.  : [self.get_tvo_loss, self.get_wake_phi_loss],          # theta optimized using tvo,  phi using wake
            'wake-wake'  : [self.get_wake_theta_loss, self.get_wake_phi_loss],   # theta optimized using wake, phi using wake
        }

        assert self.args.loss in self.losses.keys(), f"Invalid loss! Must be in {self.losses.keys()}."

        self.optimizer  = None
        self.loss = self.losses[self.args.loss]


    # ============================================================
    # ---------- Override these functions in the subclass --------
    # (the only functions that are required to be overwritten are
    # indicated with the @abc.abstractmethod decorator)
    # ============================================================

    def elbo(self):
        """
        Returns: [N, S]
        """
        self.check_internals()
        return self.log_joint() - self.log_guide()

    def record(self):
        # save stuff during the epoch using
        # self.args.wandb (overwrite in baseclass)
        if self.args.record:
            pass


    def check_internals(self):
        """Verify internal state variables have been set.
         - False means not used,
         - None means error

        """
        assert self.x is not None, "self.x not set"
        assert self.y is not None, "self.y not set"
        assert self.z is not None, "self.z not set"

    def log_joint(self):
        """
        log p(x, z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        self.check_internals()
        log_prior = self.log_prior()
        log_likelihood = self.log_likelihood()
        if log_prior.ndim == 1:
            N = self.x.shape[0]
            log_prior = mlh.spread(log_prior, N) # [1, S] -> [N, S]

        return log_prior + log_likelihood


    @abc.abstractmethod
    def set_internals(self, data, S):
        """
        Implemented by subclass

        This sets the internal state variables so all the functions work

        Raises:
            NotImplementedError: [description]
        """
        assert isinstance(data, (tuple, list)), "Data must be a tuple (X,y) or (X, )"
        return

    @abc.abstractmethod
    def log_prior(self):
        """
        log p(z) or log p(θ), depending on
        if the prior is over latent parameters
        p(z) or global parameters p(θ)

        Implemented by subclass

        Returns: [N, S] or [S]
            p(z) -> [N, S]
            p(θ) -> [S]
        Raises:
            NotImplementedError: [description]
        """
        return

    @abc.abstractmethod
    def log_likelihood(self):
        """
        log p(x|z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        return

    @abc.abstractmethod
    def log_guide(self):
        """
        log q(z|x) or log q(z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        return

    @abc.abstractmethod
    def sample_latent(self, S):
        """
        Implemented by subclass

        Note: S is in the *first* index for sample_latent,
        This is done to match pytorch's broadcasting semantics
        * can be anything. i.e. [S, N, D0, D1, ...]

        Returns: [S, *]
        Raises:
            NotImplementedError: [description]
        """
        return

    # ============================================
    # ---------- Main train function -------------
    # ============================================

    def step_epoch(self, data_loader, **kwargs):
        # wrap for progress bar if verbose flag is on
        data_loader = tqdm(data_loader) if self.args.verbose else data_loader

        train_logpx = 0
        train_elbo = 0
        for idx, data in enumerate(data_loader):
            # this loop handles the single and double objective cases
            # add new objectives into the self.losses dictionary
            for loss_func, opt in zip(self.loss, self.optimizer):
                opt.zero_grad()
                loss = loss_func(data)
                loss.backward()
                opt.step()

            # these are computed used a fixed valid_S so they are comparable across different sample sizes
            logpx, elbo = self.get_test_metrics(data, self.args.valid_S)

            # only does something if function is filled in by subclass
            self.record()

            train_logpx += logpx.item()
            train_elbo += elbo.item()

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)

        return train_logpx, train_elbo

    # ============================================
    # ---------- Main test function --------------
    # ============================================

    def test(self, data_loader, **kwargs):
        log_p_total = 0
        kl_total = 0
        num_data = 0
        data_loader = tqdm(data_loader) if self.args.verbose else data_loader
        with torch.no_grad():
            for data in data_loader:
                log_p, kl = self.get_log_p_and_kl(data, self.args.test_S)
                log_p_total += torch.sum(log_p).item()
                kl_total += torch.sum(kl).item()
                num_data += data[0].shape[0]
        return log_p_total / num_data, kl_total / num_data


    # ============================
    # ---------- Helpers ----------
    # ============================

    def init_optimizer(self):
        # Make optimizer, has to be called after subclass has been initialized
        if len(self.loss) == 2:
            theta_opt = torch.optim.Adam(
                (params for name, params in self.named_parameters() if self.args.theta_tag in name), lr=self.args.lr)
            phi_opt = torch.optim.Adam(
                (params for name, params in self.named_parameters() if self.args.phi_tag in name), lr=self.args.lr)
            self.optimizer = [theta_opt, phi_opt]
        else:
            self.optimizer = [torch.optim.Adam(self.parameters(), lr=self.args.lr)]

    # For debugging
    def show_state(self):
        for a in self:
            print(a)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


    def get_test_metrics(self, data, S):
        """
        Computes logpx, test_elbo without
        calling set_internals twice
        """
        with torch.no_grad():
            self.set_internals(data, S)
            log_weight = self.elbo()
            logpx = self.get_test_log_evidence(data, S, log_weight=log_weight)
            test_elbo = self.get_test_elbo(data, S, log_weight=log_weight)
        return logpx, test_elbo

    def get_test_log_evidence(self, data, S, log_weight=None):
        with torch.no_grad():
            if log_weight is None:
                self.set_internals(data, S)
                log_weight = self.elbo()
            log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(S)
            iwae_log_evidence = torch.mean(log_evidence)

        return iwae_log_evidence

    def get_test_elbo(self, data, S, log_weight=None):
        with torch.no_grad():
            if log_weight is None:
                self.set_internals(data, S)
                log_weight = self.elbo()
            elbo = torch.mean(log_weight)
        return elbo

    def get_log_p_and_kl(self, data, S, log_weight=None):
        with torch.no_grad():
            if log_weight is None:
                self.set_internals(data, S)
                log_weight = self.elbo()
        log_p = torch.logsumexp(log_weight, dim=1) - np.log(S)
        elbo = torch.mean(log_weight, dim=1)
        kl = log_p - elbo

        return log_p, kl

    # ============================
    # ---------- Losses ----------
    # ============================

    def get_iwae_loss(self, data):
        '''
        IWAE loss = log mean p(x,z) / q(z|x)
        '''
        assert self.reparam is True, 'Reparam must be on for iwae loss'
        self.set_internals(data, self.args.S)
        log_weight = self.elbo()
        return compute_iwae_loss(log_weight)


    def get_iwae_dreg_loss(self, data):
        assert self.reparam is True, 'Reparam must be on for iwae loss'

        #if self.args.stop_parameter_grad:
        self.enable_stop_grads()

        self.set_internals(data, self.args.S)

        log_weight = self.elbo()

        normalized_weight = util.exponentiate_and_normalize(log_weight, dim=1)

        loss = - \
            torch.mean(torch.sum(torch.pow(normalized_weight,2).detach() * log_weight, 1), 0)
        return loss


    def get_elbo_loss(self, data):
        assert self.reparam is True, 'Reparam must be on for elbo loss'
        self.set_internals(data, self.args.S)

        log_weight = self.elbo()
        train_elbo = torch.mean(log_weight)

        loss = -train_elbo
        return loss

    def get_reinforce_loss(self, data):
        assert self.reparam is False, 'Reparam must be off for reinforce loss'
        self.set_internals(data, self.args.S)

        log_weight = self.elbo()
        log_q = self.log_guide()

        reinforce = log_weight.detach() * log_q + log_weight

        loss = -torch.mean(reinforce)
        return loss

    def get_tvo_loss(self, data):
        if self.args.loss != 'tvo_reparam':
            assert self.reparam is False, 'Reparam must be off for TVO loss'

        self.set_internals(data, self.args.S)

        if self.args.per_sample:
            self.args.partition = self.args.partition_scheduler(self, self.args)

        log_weight = self.elbo()
        log_joint = self.log_joint()
        log_guide = self.log_guide()
        loss = compute_tvo_loss(log_weight, log_joint, log_guide, self.args)

        return loss

    def get_tvo_reparam_loss(self, data):
        assert self.reparam is True, 'Reparam must be ON for TVO reparam'

        # used to prevent score function gradients
        self.enable_stop_grads()

        self.set_internals(data, self.args.S)

        if self.args.per_sample or self.args.per_batch:
            self.args.partition = self.args.partition_scheduler(self, self.args)

        log_weight = self.elbo()
        log_joint = self.log_joint()
        log_guide = self.log_guide()

        loss = compute_tvo_reparam_loss(log_weight, log_joint, log_guide, self.args)

        return loss


    def get_wake_theta_loss(self, data):
        """Scalar that we call .backward() on and step the optimizer.

        Args:
            generative_model: models.GenerativeModel object
            inference_network: models.InferenceNetwork object
            obs: tensor of shape [batch_size]
            num_particles: int

        Returns:
            loss: scalar that we call .backward() on and step the optimizer.
            elbo: average elbo over data
        """
        assert self.reparam is False, 'Reparam must be off for wake_theta_loss'
        self.set_internals(data, self.args.S)

        log_weight = self.elbo()
        return compute_wake_theta_loss(log_weight)

    def get_wake_phi_loss(self, data):
        """
        Args:
            generative_model: models.GenerativeModel object
            inference_network: models.InferenceNetwork object
            obs: tensor of shape [batch_size]
            num_particles: int

        Returns:
            loss: scalar that we call .backward() on and step the optimizer.
        """
        assert self.reparam is False, 'Reparam must be off for wake_phi_loss'
        self.set_internals(data, self.args.S)

        log_weight = self.elbo()
        log_q = self.log_guide()
        return compute_wake_phi_loss(log_weight, log_q)

    def get_sleep_phi_loss(self, data):
        """Returns:
        loss: scalar that we call .backward() on and step the optimizer.
        """
        assert self.reparam is False, 'Reparam must be off for sleep_loss'
        self.set_internals(data, self.args.S)
        log_q = self.log_guide()
        return -torch.mean(log_q)

    def get_vimco_loss(self, data):
        """Almost twice faster version of VIMCO loss (measured for batch_size = 24,
        num_particles = 1000). Inspired by Adam Kosiorek's implementation.

        Args:
            generative_model: models.GenerativeModel object
            inference_network: models.InferenceNetwork object
            obs: tensor of shape [batch_size]
            num_particles: int

        Returns:

            loss: scalar that we call .backward() on and step the optimizer.
            elbo: average elbo over data
        """
        assert self.reparam is False, 'Reparam must be off for vimco_loss'
        self.set_internals(data, self.args.S)

        # assert self.reparam is True, 'Reparam must be on for wake_phi_loss'
        log_weight = self.elbo()
        log_q = self.log_guide()
        return compute_vimco_loss(log_weight, log_q)

    def get_concrete_loss(self, data):
        raise NotImplementedError

    def get_relax_loss(self, data):
        raise NotImplementedError
