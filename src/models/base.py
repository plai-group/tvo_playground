from torch import nn
from src import ml_helpers as mlh
from collections import defaultdict
import numpy as np
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

        self.hist = defaultdict(list)
        self.record_results = defaultdict(mlh.AverageMeter)

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

        if self.args.loss in assertions.DUAL_LOSSES:
            self.dual_objective = True
        else:
            self.dual_objective = False

        self.optimizer  = None
        self.optimizer_phi_only   = None
        self.optimizer_theta_only = None


    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    # For debugging
    def show_state(self):
        for a in self:
            print(a)

    def elbo(self):
        """
        Returns: [N, S]
        """
        self.check_internals()
        return self.log_joint() - self.log_guide()

    def set_internals(self, data, S):
        """
        Implemented by subclass

        This sets the internal state variables so all the functions work

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

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
        prior = self.log_prior()
        likelihood = self.log_likelihood()
        if prior.ndim == 1:
            N = self.x.shape[0]
            prior = (1 / N) * prior.unsqueeze(0).repeat(N, 1)
            return likelihood + (1 / self.args.batch_size) * prior
        else:
            return prior + likelihood

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
        raise NotImplementedError

    def log_likelihood(self):
        """
        log p(x|z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def log_guide(self):
        """
        log q(z|x) or log q(z)
        Implemented by subclass

        Returns: [N, S]
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def init_optimizer(self):
        # Make optimizer
        if self.dual_objective:
            self.optimizer_phi_only = torch.optim.Adam(
                (params for name, params in self.named_parameters() if self.args.phi_tag in name), lr=self.args.lr)
            self.optimizer_theta_only = torch.optim.Adam(
                (params for name, params in self.named_parameters() if self.args.theta_tag in name), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)


    # ============================
    # ---------- Helpers ----------
    # ============================

    def get_test_log_evidence(self, data, S):
        with torch.no_grad():
            self.set_internals(data, S)
            log_weight = self.elbo()
            log_evidence = torch.logsumexp(log_weight, dim=1) - np.log(S)
            iwae_log_evidence = torch.mean(log_evidence)

        return iwae_log_evidence

    def get_test_elbo(self, data, S):
        with torch.no_grad():
            self.set_internals(data, S)
            log_weight = self.elbo()
            elbo = torch.mean(log_weight)
        return elbo

    def get_log_p_and_kl(self, data, S):
        self.set_internals(data, S)
        log_weight = self.elbo()

        log_p = torch.logsumexp(log_weight, dim=1) - np.log(S)
        elbo = torch.mean(log_weight, dim=1)
        kl = log_p - elbo

        return log_p, kl

    def record_artifacts(self, _run):
        # need to find a better way to do this
        pass

    def save_record(self):
        # need to find a better way to do this
        pass


    def train(self, data_loader):
        if self.dual_objective:
            return self.train_dual_objectives(data_loader)
        else:
            return self.train_single_objective(data_loader)

    def train_single_objective(self, data_loader):
        train_logpx = 0
        train_elbo = 0
        for idx, data in enumerate(data_loader):
            self.optimizer.zero_grad()

            loss, logpx, elbo = self.forward(data)

            loss.backward()
            self.optimizer.step()

            if self.args.record:
                self.record_stats()

            train_logpx += logpx.item()
            train_elbo += elbo.item()

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)

        if self.args.record:
            self.save_record()

        return train_logpx, train_elbo

    def train_dual_objectives(self, data_loader):
        train_logpx = 0
        train_elbo = 0
        for idx, data in enumerate(data_loader):
            self.optimizer_phi_only.zero_grad()
            self.optimizer_theta_only.zero_grad()

            if self.args.loss == 'tvo_reparam': # p optimized using tvo
                wake_theta_loss = self.get_tvo_loss(data)
            elif self.args.loss == 'iwae_dreg': # p optimized using IWAE (DReG update is only for q)
                wake_theta_loss = self.get_iwae_loss(data)
            elif self.args.loss in ['wake-wake', 'wake-sleep']:
                wake_theta_loss = self.get_wake_theta_loss(data)
            else:
                raise ValueError(f"{self.args.loss} is an invalid loss")

            wake_theta_loss.backward()
            self.optimizer_theta_only.step()

            if self.args.loss in ['tvo_reparam']:
                sleep_phi_loss = self.get_tvo_reparam_loss(data)
                sleep_phi_loss.backward()
            elif self.args.loss == 'iwae_dreg':
                sleep_phi_loss = self.get_iwae_dreg_loss(data)
                sleep_phi_loss.backward()
            elif self.args.loss == 'wake-sleep':
                sleep_phi_loss = self.get_sleep_phi_loss(data)
                sleep_phi_loss.backward()
            elif self.args.loss == 'wake-wake':
                wake_phi_loss = self.get_wake_phi_loss(data)
                wake_phi_loss.backward()
            else:
                raise ValueError(f"{self.args.loss} is an invalid loss")

            self.optimizer_phi_only.step()

            logpx = self.get_test_log_evidence(data, self.args.valid_S)
            elbo = self.get_test_elbo(data, self.args.valid_S)

            if self.args.record:
                self.record_stats()

            train_logpx += logpx.item()
            train_elbo += elbo.item()

        if self.args.record:
            self.save_record()

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)

        return train_logpx, train_elbo

    def evaluate_model_and_inference_network(self, data_loader):
        log_p_total = 0
        kl_total = 0
        num_data = 0
        with torch.no_grad():
            for data in iter(data_loader):
                log_p, kl = self.get_log_p_and_kl(data, self.args.test_S)
                log_p_total += torch.sum(log_p).item()
                kl_total += torch.sum(kl).item()
                num_data += data[0].shape[0]
        return log_p_total / num_data, kl_total / num_data

    def evaluate_model(self, data_loader):
        log_px = 0
        with torch.no_grad():
            for idx, data in enumerate(data_loader):
                log_px += self.get_test_log_evidence(data, self.args.test_S)
        return log_px / len(data_loader)

    def record_stats(self):
        '''
            Records (across β) : expectation / variance / 3rd / 4th derivatives
                curvature, IWAE β estimator
                intermediate TVO integrals (WIP)
        '''

        '''Possibility of different, standardized partition for evaluation?
            - may run with record_partition specified or overall arg'''

        # Always use validation sample size
        S = self.args.valid_S

        # Always use same partition grid for comparability
        partition = mlh.tensor(np.linspace(0, 1.0, 101), args=self.args)

        with torch.no_grad():
            log_iw = self.elbo().unsqueeze(-1) if len(self.elbo().shape) < 3 else self.elbo()

            heated_log_weight = log_iw * partition

            snis = util.exponentiate_and_normalize(heated_log_weight, dim=1)

            # Leaving open possibility of addl calculations on batch dim (mean = False)
            tvo_expectations = util.calc_exp(
                log_iw, partition, snis=snis, all_sample_mean=True)
            tvo_vars = util.calc_var(
                log_iw, partition, snis=snis, all_sample_mean=True)
            tvo_thirds = util.calc_third(
                log_iw, partition, snis=snis, all_sample_mean=True)
            tvo_fourths = util.calc_fourth(
                log_iw, partition, snis=snis, all_sample_mean=True)

            curvature = tvo_thirds/(torch.pow(1+torch.pow(tvo_vars, 2), 1.5))
            iwae_beta = torch.mean(torch.logsumexp(
                heated_log_weight, dim=1) - np.log(S), axis=0)

            # Using average meter
            # torch.mean(tvo_expectations, dim=0)
            self.record_results['tvo_exp'].step(tvo_expectations.cpu())
            self.record_results['tvo_var'].step(tvo_vars.cpu())
            self.record_results['tvo_third'].step(tvo_thirds.cpu())
            self.record_results['tvo_fourth'].step(tvo_fourths.cpu())

            # per sample curvature by beta (gets recorded as mean over batches)
            self.record_results['curvature'].step(curvature.cpu())
            # [K] length vector of MC estimators of log Z_β
            self.record_results['iwae_beta'].step(iwae_beta.cpu())

    # ============================
    # ---------- Losses ----------
    # ============================

    def forward(self, data):
        assert isinstance(data, (tuple, list)), "Data must be a tuple (X,y) or (X, )"

        if self.args.loss == 'reinforce':
            loss = self.get_reinforce_loss(data)
        elif self.args.loss == 'elbo':
            loss = self.get_elbo_loss(data)
        elif self.args.loss == 'iwae':
            loss = self.get_iwae_loss(data)
        elif self.args.loss == 'tvo':
            loss = self.get_tvo_loss(data)
        elif self.args.loss == 'vimco':
            loss = self.get_vimco_loss(data)
        else:
            raise ValueError(f"{self.args.loss} is an invalid loss")

        logpx = self.get_test_log_evidence(data, self.args.valid_S)
        test_elbo = self.get_test_elbo(data, self.args.valid_S)
        return loss, logpx, test_elbo


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
