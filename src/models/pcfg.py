import torch
import torch.nn as nn
import src.pcfg_util as util
from torch.distributions import *
from src.models.base import ProbModelBaseClass
import numpy as np


class GenerativeModel(nn.Module):
    def __init__(self, grammar, production_probs_init=None, max_depth=30):
        super(GenerativeModel, self).__init__()
        self.grammar = grammar
        if self.grammar['name'] == 'polynomial':
            self.xs = torch.linspace(-10, 10, 100)
        if production_probs_init is None:
            self.production_logits = nn.ParameterDict({
                k: nn.Parameter(torch.randn((len(v),)))
                for k, v in grammar['productions'].items()})
        else:
            self.production_logits = nn.ParameterDict({
                k: nn.Parameter(torch.log(v))
                for k, v in production_probs_init.items()})
        self.max_depth = max_depth

    def sample_tree(self, symbol=None, depth=0):
        """Sample tree from prior.

        Args: start symbol
        Returns: list of lists or string
        """

        if symbol is None:
            symbol = self.grammar['start_symbol']

        if symbol in self.grammar['terminals']:
            return symbol
        elif depth > self.max_depth:
            return symbol
        else:
            dist = Categorical(logits=self.production_logits[symbol])
            production_index = dist.sample().detach()
            production = self.grammar['productions'][symbol][production_index]
            return [symbol] + \
                [self.sample_tree(s, depth=depth + 1) for s in production]

    def sample_tree_and_obs(self):
        """Samples a (tree, obs) tuple from prior."""

        tree = self.sample_tree()
        if self.grammar['name'] == 'astronomers':
            sentence = util.get_leaves(tree)
            obs = sentence
        elif self.grammar['name'] == 'polynomial':
            ys = util.eval_polynomial(tree, self.xs)
            obs = ys
        return tree, obs

    def sample_obs(self):
        """Samples obs from prior."""

        return self.sample_tree_and_obs()[1]

    def get_tree_log_prob(self, tree):
        """Log probability of tree.

        Args:
            tree: list of lists or string

        Returns: scalar tensor
        """

        if isinstance(tree, list):
            non_terminal = tree[0]
            subtrees = tree[1:]
            production = [util.get_root(subtree) for subtree in subtrees]
            production_index = util.get_production_index(
                non_terminal, production, self.grammar['productions'])
            dist = Categorical(logits=self.production_logits[non_terminal])
            log_prob = dist.log_prob(torch.tensor(production_index))
            subtree_log_probs = [self.get_tree_log_prob(subtree)
                                 for subtree in subtrees]
            return log_prob + sum(subtree_log_probs)
        else:
            return torch.zeros(())

    def get_sentence_log_likelihood(self, sentence, tree):
        """Minus ABC distance instead of log p(sentence | tree). ABC distance
        is the Levenshtein distance.

        Args:
            sentence: list of strings
            tree: list of lists or string

        Returns: scalar tensor"""

        sentence_from_tree = util.get_leaves(tree)
        levenshtein_distance = torch.tensor(
            util.get_levenshtein_distance(sentence_from_tree, sentence,
                                          self.grammar['terminals']),
            dtype=torch.float)
        # if levenshtein_distance.item() == 0:
        #     return levenshtein_distance
        # else:
        #     return torch.tensor(float('-inf'))
        # return -(torch.exp(levenshtein_distance) - 1)
        # return -levenshtein_distance
        return -levenshtein_distance**2

    def get_polynomial_log_likelihood(self, ys, tree):
        """Minus ABC distance instead of log p(ys | tree, xs) where xs is
            torch.linspace(-10, 10, 100). ABC distance is log(1 + mse).

        Args:
            ys: torch.tensor of shape [100]
            tree: list of lists or string

        Returns: -log(1 + mse(ys, eval(tree))); scalar tensor
        """

        return -torch.log(
            1 + util.mse(ys, util.eval_polynomial(tree, self.xs)))

    def get_log_prob(self, tree, obs, sum_prior_and_likelihood=True):
        """Joint log probability p(obs, tree).

        Args:
            tree: list of lists or string
            obs: sentence (list of strings) or ys (torch.tensor of shape [100])

        Returns: scalar tensor
        """

        if self.grammar['name'] == 'astronomers':
            sentence = obs
            if sum_prior_and_likelihood:
                return self.get_tree_log_prob(tree) + \
                    self.get_sentence_log_likelihood(sentence, tree)
            else:
                return self.get_tree_log_prob(tree), self.get_sentence_log_likelihood(sentence, tree)

            # The following is the non-ABC version for which p(sentence | tree)
            # is 1 if tree's leaves match the sentence and 0 otherwise
            #
            # if util.get_leaves(tree) == sentence:
            #     return self.get_tree_log_prob(tree)
            # else:
            #     return torch.tensor(float('-inf'))
        elif self.grammar['name'] == 'polynomial':
            ys = obs
            if sum_prior_and_likelihood:
                return self.get_tree_log_prob(tree) + \
                    self.get_polynomial_log_likelihood(ys, tree)
            else:
                return self.get_tree_log_prob(tree), self.get_polynomial_log_likelihood(ys, tree)


class InferenceNetwork(nn.Module):
    def __init__(self, grammar, obs_embedding_dim=100,
                 inference_hidden_dim=100, max_depth=30):
        super(InferenceNetwork, self).__init__()
        self.grammar = grammar
        self.obs_embedding_dim = obs_embedding_dim
        self.inference_hidden_dim = inference_hidden_dim
        self.max_depth = max_depth
        self.sample_address_embedding_dim = len(grammar['non_terminals'])
        self.word_embedding_dim = len(self.grammar['terminals'])

        if grammar['name'] == 'astronomers':
            self.sentence_embedder_gru = nn.GRU(
                input_size=self.word_embedding_dim,
                hidden_size=self.obs_embedding_dim,
                num_layers=1)
        elif grammar['name'] == 'polynomial':
            self.xs = torch.linspace(-10, 10, 100)
            self.gray_embedder_cnn = nn.Sequential(
                nn.Conv2d(1, 20, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(20, 20, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(20, 20, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(20, 10, 3),
                nn.ReLU())
            self.gray_embedder_mlp = nn.Sequential(
                nn.Linear(640, 320),
                nn.ReLU(),
                nn.Linear(320, 160),
                nn.ReLU(),
                nn.Linear(160, obs_embedding_dim))
        self.sample_embedding_dim = max(
            [len(v) for _, v in self.grammar['productions'].items()])
        self.inference_gru = nn.GRUCell(
            input_size=self.obs_embedding_dim + self.sample_embedding_dim
            + self.sample_address_embedding_dim,
            hidden_size=self.inference_hidden_dim)
        self.proposal_layers = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(inference_hidden_dim, 50),
                             nn.ReLU(),
                             nn.Linear(50, 25),
                             nn.ReLU(),
                             nn.Linear(25, len(v)))
            for k, v in grammar['productions'].items()})

    def get_sentence_embedding(self, sentence):
        """Args:
            sentence: list of strings

        Returns: tensor of shape [obs_embedding_dim]
        """

        output, _ = self.sentence_embedder_gru(util.sentence_to_one_hots(
            sentence, self.grammar['terminals']).unsqueeze(1))
        return output[-1][0]

    def get_ys_embedding(self, ys):
        """Args:
            ys: tensor of shape [100]

        Returns: tensor of shape [obs_embedding_dim]
        """

        gray = util.xsys2gray(self.xs, ys)
        input_to_mlp = self.gray_embedder_cnn(
            gray.view(1, 1, 100, 100)).view(-1).squeeze(0)
        return self.gray_embedder_mlp(input_to_mlp).squeeze(0)

    def get_obs_embedding(self, obs):
        """Args:
            obs: sentence (list of strings) or ys (torch.tensor of shape [100])

        Returns: tensor of shape [obs_embedding_dim]
        """

        if self.grammar['name'] == 'astronomers':
            sentence = obs
            return self.get_sentence_embedding(sentence)
        elif self.grammar['name'] == 'polynomial':
            ys = obs
            return self.get_ys_embedding(ys)

    def get_logits_from_inference_gru_output(self, inference_gru_output,
                                             non_terminal):
        """Args:
            inference_gru_output: tensor of shape [inference_hidden_dim]
            non_terminal: string

        Returns: logits for Categorical distribution
        """

        input_ = inference_gru_output.unsqueeze(0)
        return self.proposal_layers[non_terminal](input_).squeeze(0)

    def get_sample_embedding(self, production_index):
        """Args: int

        Returns: one hot vector of shape [sample_embedding_dim]
        """
        return util.one_hot(torch.tensor([production_index]),
                            self.sample_embedding_dim)[0]

    def get_inference_gru_output(self, obs_embedding,
                                 previous_sample_embedding,
                                 sample_address_embedding, inference_hidden):
        """Args:
            obs_embedding: tensor [obs_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            sample_address_embedding: tensor [sample_embedding_address_dim]
            inference_hidden: tensor [inference_hidden_dim]

        Returns: tensor [inference_hidden_dim]
        """

        return self.inference_gru(
            torch.cat([obs_embedding,
                       previous_sample_embedding,
                       sample_address_embedding]).unsqueeze(0),
            inference_hidden.unsqueeze(0)).squeeze(0)

    def get_tree_log_prob(self, tree, obs_embedding=None,
                          previous_sample_embedding=None,
                          inference_hidden=None, obs=None):
        """Log probability of tree given obs.

        Args:
            tree: list or string
            obs_embedding: tensor [obs_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            inference_hidden: tensor [inference_hidden_dim]
            obs: sentence (list of strings) or ys (torch.tensor of shape [100])

        Returns: log_prob (scalar tensor)"""

        if obs_embedding is None:
            obs_embedding = self.get_obs_embedding(obs)

        if previous_sample_embedding is None:
            previous_sample_embedding = torch.zeros(
                (self.sample_embedding_dim,))

        if inference_hidden is None:
            inference_hidden = torch.zeros((self.inference_hidden_dim,))

        if isinstance(tree, list):
            non_terminal = tree[0]
            sample_address_embedding = util.get_sample_address_embedding(
                non_terminal, self.grammar['non_terminals'])
            inference_gru_output = self.get_inference_gru_output(
                obs_embedding, previous_sample_embedding,
                sample_address_embedding, inference_hidden)

            subtrees = tree[1:]
            production = [util.get_root(subtree) for subtree in subtrees]
            production_index = util.get_production_index(
                non_terminal, production, self.grammar['productions'])
            sample_embedding = self.get_sample_embedding(production_index)
            logits = self.get_logits_from_inference_gru_output(
                inference_gru_output, non_terminal)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(production_index))
            subtree_log_probs = [
                self.get_tree_log_prob(subtree, obs_embedding,
                                       sample_embedding, inference_gru_output)
                for subtree in subtrees]
            return log_prob + sum(subtree_log_probs)
        else:
            return torch.zeros(())

    def sample_tree(self, symbol=None, obs_embedding=None,
                    previous_sample_embedding=None, inference_hidden=None,
                    obs=None, depth=0):
        """Samples a tree given a obs and a start symbol (can be terminal
            or non-terminal).

        Args:
            symbol: string
            obs_embedding: tensor [obs_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            inference_hidden: tensor [inference_hidden_dim]
            obs: sentence (list of strings) or ys (torch.tensor of shape [100])

        Returns: tree
        """

        if symbol is None:
            symbol = self.grammar['start_symbol']

        if obs_embedding is None:
            obs_embedding = self.get_obs_embedding(obs)

        if previous_sample_embedding is None:
            previous_sample_embedding = torch.zeros(
                (self.sample_embedding_dim,))

        if inference_hidden is None:
            inference_hidden = torch.zeros((self.inference_hidden_dim,))

        if symbol in self.grammar['terminals']:
            return symbol
        elif depth > self.max_depth:
            return symbol
        else:
            sample_address_embedding = util.get_sample_address_embedding(
                symbol, self.grammar['non_terminals'])
            inference_gru_output = self.get_inference_gru_output(
                obs_embedding, previous_sample_embedding,
                sample_address_embedding, inference_hidden)
            logits = self.get_logits_from_inference_gru_output(
                inference_gru_output, symbol)
            dist = Categorical(logits=logits)
            production_index = dist.sample().detach()
            sample_embedding = self.get_sample_embedding(production_index)
            production = self.grammar['productions'][symbol][production_index]

            return [symbol] + [
                self.sample_tree(s, obs_embedding, sample_embedding,
                                 inference_gru_output, depth=depth + 1)
                for s in production]

    def sample_tree_relax(self, symbol=None, obs_embedding=None,
                          previous_sample_embedding=None,
                          inference_hidden=None, obs=None, depth=0):
        """Samples a tree given a obs and a start symbol (can be terminal
            or non-terminal).

        Args:
            symbol: string
            obs_embedding: tensor [obs_embedding_dim]
            previous_sample_embedding: tensor [sample_embedding_dim]
            inference_hidden: tensor [inference_hidden_dim]
            obs: sentence (list of strings) or ys (torch.tensor of shape [100])

        Returns:
            tree: e.g.
                ['S', ['NP', 'astronomers'],
                      ['VP', ['V' 'saw'],
                             ['NP' 'stars']]]
                or 'stars'
            tree_aux: e.g.
                [[0.5], [[.9, 1., .2, .1, -.1, .1], None],
                        [[-0.3 0.8], [[0.3], None]
                                     [[.9, -.1, .2, .1, 1., .1], None]]]
                or None
            tree_aux_tilde: similar to tree_aux
        """

        if symbol is None:
            symbol = self.grammar['start_symbol']

        if obs_embedding is None:
            obs_embedding = self.get_obs_embedding(obs)

        if previous_sample_embedding is None:
            previous_sample_embedding = torch.zeros(
                (self.sample_embedding_dim,))

        if inference_hidden is None:
            inference_hidden = torch.zeros((self.inference_hidden_dim,))

        if symbol in self.grammar['terminals']:
            return symbol, None, None
        elif depth > self.max_depth:
            return symbol, None, None
        else:
            sample_address_embedding = util.get_sample_address_embedding(
                symbol, self.grammar['non_terminals'])
            inference_gru_output = self.get_inference_gru_output(
                obs_embedding, previous_sample_embedding,
                sample_address_embedding, inference_hidden)
            logits = self.get_logits_from_inference_gru_output(
                inference_gru_output, symbol)
            oh_production_index, production_index_aux, \
                production_index_aux_tilde = util.sample_relax(logits=logits)
            production_index = torch.argmax(oh_production_index)
            sample_embedding = self.get_sample_embedding(production_index)
            production = self.grammar['productions'][symbol][production_index]

            tree = [symbol]
            tree_aux = [production_index_aux]
            tree_aux_tilde = [production_index_aux_tilde]
            for s in production:
                subtree, subtree_aux, subtree_aux_tilde = \
                    self.sample_tree_relax(
                        s, obs_embedding, sample_embedding,
                        inference_gru_output, depth=depth + 1)
                tree.append(subtree)
                tree_aux.append(subtree_aux)
                tree_aux_tilde.append(subtree_aux_tilde)
            return tree, tree_aux, tree_aux_tilde


class ControlVariate(nn.Module):
    def __init__(self, grammar, obs_embedding_dim=100,
                 tree_obs_embedding_dim=100):
        super(ControlVariate, self).__init__()
        self.grammar = grammar
        self.obs_embedding_dim = obs_embedding_dim
        self.word_embedding_dim = len(self.grammar['terminals'])
        self.tree_obs_embedding_dim = tree_obs_embedding_dim
        self.obs_embedder_gru = nn.GRU(
            input_size=self.word_embedding_dim,
            hidden_size=self.obs_embedding_dim,
            num_layers=1)
        self.sample_address_embedding_dim = len(grammar['non_terminals'])
        self.sample_embedding_dim = max(
            [len(v) for _, v in grammar['productions'].items()])
        self.tree_obs_embedder_gru = nn.GRUCell(
            input_size=self.obs_embedding_dim + self.sample_embedding_dim +
            self.sample_address_embedding_dim,
            hidden_size=tree_obs_embedding_dim)
        self.tree_obs_mlp = nn.Sequential(
            nn.Linear(tree_obs_embedding_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1))

    def get_obs_embedding(self, obs):
        """Args:
            obs: list of strings

        Returns: tensor of shape [obs_embedding_dim]
        """

        output, _ = self.obs_embedder_gru(util.sentence_to_one_hots(
            obs, self.grammar['terminals']).unsqueeze(1))
        return output[-1][0]

    def get_tree_obs_gru_output(self, obs_embedding, sample_embedding,
                                sample_address_embedding, tree_obs_hidden):
        """Args:
            obs_embedding: tensor [obs_embedding_dim]
            sample_embedding: tensor [sample_embedding_dim]
            sample_address_embedding: tensor [sample_embedding_address_dim]
            tree_obs_hidden: tensor [tree_obs_embedding_dim]

        Returns: tensor of shape [tree_obs_embedding_dim]
        """
        return self.tree_obs_embedder_gru(
            torch.cat([obs_embedding,
                       sample_embedding,
                       sample_address_embedding]).unsqueeze(0),
            tree_obs_hidden.unsqueeze(0)).squeeze(0)

    def get_tree_obs_embedding(self, tree, tree_aux, obs_embedding):
        """Args:
            tree: e.g.
                ['S', ['NP', 'astronomers'],
                      ['VP', ['V' 'saw'],
                             ['NP' 'stars']]]
                or 'stars'
            tree_aux: e.g.
                [[0.5], [[.9, 1., .2, .1, -.1, .1], None],
                        [[-0.3 0.8], [[0.3], None]
                                     [[.9, -.1, .2, .1, 1., .1], None]]]
                or None
            obs_embedding: tensor of shape [obs_embedding_dim]

        Returns: tensor of shape [tree_obs_embedding_dim]
        """

        if isinstance(tree, list):
            non_terminal = tree[0]
            sample_address_embedding = util.get_sample_address_embedding(
                non_terminal, self.grammar['non_terminals'])
            sample_embedding = util.pad_zeros(tree_aux[0],
                                              self.sample_embedding_dim)
            subtrees = tree[1:]
            subtrees_aux = tree_aux[1:]
            tree_obs_hidden = 0
            for subtree, subtree_aux in zip(subtrees, subtrees_aux):
                tree_obs_hidden += self.get_tree_obs_embedding(
                    subtree, subtree_aux, obs_embedding)
            return self.get_tree_obs_gru_output(
                obs_embedding, sample_embedding, sample_address_embedding,
                tree_obs_hidden)
        else:
            return torch.zeros((self.tree_obs_embedding_dim,))

    def control_variate_single(self, tree, tree_aux, obs_embedding):
        """Args:
            tree: e.g.
                ['S', ['NP', 'astronomers'],
                      ['VP', ['V' 'saw'],
                             ['NP' 'stars']]]
                or 'stars'
            tree_aux: e.g.
                [[0.5], [[.9, 1., .2, .1, -.1, .1], None],
                        [[-0.3 0.8], [[0.3], None]
                                     [[.9, -.1, .2, .1, 1., .1], None]]]
                or None
            obs_embedding: tensor of shape [obs_embedding_dim]

        Returns: scalar tensor
        """
        return self.tree_obs_mlp(self.get_tree_obs_embedding(
            tree, tree_aux, obs_embedding).unsqueeze(0)).squeeze(0)

    def forward(self, trees, trees_aux, obs_embeddings):
        """Args:
            trees_aux: list of lists of shape [num_obs, num_particles] where
                each element is either a tree_aux or tree_aux_tilde
            obs_embeddings: list of tensors of length num_obs where each tensor
                is of shape [obs_embedding_dim]

        Returns: tensor of shape [num_obs]
        """

        num_obs = len(obs_embeddings)
        num_particles = len(trees_aux[0])
        c = torch.zeros(num_obs, num_particles)
        for obs_idx in range(num_obs):
            for particle_idx in range(num_particles):
                c[obs_idx, particle_idx] = self.control_variate_single(
                    trees[obs_idx][particle_idx],
                    trees_aux[obs_idx][particle_idx], obs_embeddings[obs_idx])
        return torch.logsumexp(c, dim=1) - np.log(num_particles)



class PCFG(ProbModelBaseClass):
    def __init__(self, grammar, args, **kwargs):
        D = None # trees have no fixed dimension
        super(PCFG, self).__init__(D, args)
        self.generative_model = GenerativeModel(grammar, **kwargs)
        self.inference_network = InferenceNetwork(grammar, **kwargs)

        self._log_q = None
        self._log_prior = None
        self._log_likelihood = None


    def set_internals(self, data, S):
        self.x = data[0]
        self.y = False # unsupervised
        self.z = False # we compute log_prior, log_likelihood, log_guide directly below

        log_q = torch.zeros(len(self.x), S)
        log_prior = torch.zeros(len(self.x), S)
        log_likelihood = torch.zeros(len(self.x), S)

        # this is painful, batching difficult b/c of different length trees.
        # therefore iterate once and save log_prior, log_guide, log_likelihood
        # manually
        for obs_idx, obs in enumerate(self.x):
            for particle_idx in range(S):
                tree = self.inference_network.sample_tree(obs=obs)
                log_q_ = self.inference_network.get_tree_log_prob(tree, obs=obs)
                log_prior_, log_likelihood_ = self.generative_model.get_log_prob(tree, obs, sum_prior_and_likelihood=False)
                log_q[obs_idx, particle_idx] = log_q_
                log_prior[obs_idx, particle_idx] = log_prior_
                log_likelihood[obs_idx, particle_idx] = log_likelihood_

        self._log_q  = log_q
        self._log_prior  = log_prior
        self._log_likelihood  = log_likelihood

        self.check_internals()

    def check_internals(self):
        super().check_internals()
        assert self._log_q is not None, "self._log_q not set"
        assert self._log_prior is not None, "self._log_prior not set"
        assert self._log_likelihood is not None, "self._log_likelihood not set"

    def sample_latent(self, S, sample=True):
        raise ValueError("Sample latent not used in PCFG")

    def log_prior(self):
        return self._log_prior

    def log_guide(self):
        return self._log_q

    def log_likelihood(self):
        return self._log_likelihood

    # These functions are overwritten to avoid repeated calls to set_internal which is expensive
    def train_epoch_single_objective(self, data_loader, optimizer, epoch=None):
        train_logpx = 0
        train_elbo = 0
        train_tvo_log_evidence = 0

        for idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            loss, logpx, elbo, tvo_log_evidence = self.forward(data)
            loss.backward()
            optimizer.step()

            train_logpx += logpx.item()
            train_elbo += elbo.item()
            train_tvo_log_evidence+=tvo_log_evidence.item()

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)
        train_tvo_log_evidence = train_tvo_log_evidence/len(data_loader)

        self.evaluate_pq(data_loader, epoch)

        return train_logpx, train_elbo, train_tvo_log_evidence


    def train_epoch_dual_objectives(self, data_loader, optimizer_phi, optimizer_theta, epoch=None):
        train_logpx = 0
        train_elbo = 0
        train_tvo_log_evidence = 0
        for idx, data in enumerate(data_loader):
            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()
            if self.args.loss == 'tvo-sleep':
                wake_theta_loss = self.get_tvo_loss(data)
            else:
                wake_theta_loss = self.get_wake_theta_loss(data)
            wake_theta_loss.backward()
            optimizer_theta.step()

            optimizer_phi.zero_grad()
            optimizer_theta.zero_grad()

            if self.args.loss in ['wake-sleep', 'tvo-sleep']:
                sleep_phi_loss = self.get_sleep_phi_loss()
                sleep_phi_loss.backward()
            elif self.args.loss in ['wake-wake']:
                wake_phi_loss = self.get_wake_phi_loss(data)
                wake_phi_loss.backward()
            else:
                raise ValueError(
                    "{} is an invalid loss".format(self.args.loss))

            optimizer_phi.step()

            with torch.no_grad():
                log_weight = self.elbo()
                logpx = self.get_test_log_evidence(data, self.args.S, log_weight=log_weight)
                elbo = self.get_test_elbo(data, self.args.S, log_weight=log_weight)
                tvo_log_evidence = self.get_tvo_log_evidence(data, self.args.S, log_weight=log_weight)

            train_logpx += logpx.item()
            train_elbo += elbo.item()
            train_tvo_log_evidence+=tvo_log_evidence.item()

        train_logpx = train_logpx / len(data_loader)
        train_elbo = train_elbo / len(data_loader)
        train_tvo_log_evidence=train_tvo_log_evidence/len(data_loader)

        self.evaluate_pq(data_loader, epoch)

        return train_logpx, train_elbo, train_tvo_log_evidence

    def forward(self, data):
        assert isinstance(data, (tuple, list)), "Data must be a tuple (X,y) or (X, )"

        if self.args.loss == 'reinforce':
            loss = self.get_reinforce_loss(data)
        elif self.args.loss == 'thermo' or self.args.loss == 'tvo':
            loss = self.get_tvo_loss(data)
        else:
            raise ValueError("{} is an invalid loss".format(self.args.loss))

        # This functions is overwritten to avoid repeated calls to set_internal which is expensive
        with torch.no_grad():
            log_weight = self.elbo()
            logpx = self.get_test_log_evidence(data, self.args.S, log_weight=log_weight)
            elbo = self.get_test_elbo(data, self.args.S, log_weight=log_weight)
            tvo_log_evidence = self.get_tvo_log_evidence(data, self.args.S, log_weight=log_weight)

        return loss, logpx, elbo, tvo_log_evidence

    def evaluate_pq(self, data_loader, epoch):
        true_generative_model = data_loader.dataset.true_generative_model

        metrics = {
            "p_error":util.get_p_error(true_generative_model, self.generative_model),
            "q_error_to_true":util.get_q_error(true_generative_model, self.inference_network),
            "q_error_to_model":util.get_q_error(self.generative_model, self.inference_network)
            }

        for k, v in metrics.items():
            self.args._run.log_scalar(k, float(v), epoch)

        loss_string = " ".join(("{}: {:.4f}".format(*i) for i in metrics.items()))
        print(f"Epoch: {epoch} - {loss_string}")

    def get_sleep_phi_loss(self):
        """Returns:
            loss: scalar that we call .backward() on and step the optimizer.
        """
        log_q_sum = 0
        for _ in range(self.args.S):
            tree, obs = self.generative_model.sample_tree_and_obs()
            log_q = self.inference_network.get_tree_log_prob(tree, obs=obs)
            log_q_sum = log_q_sum + log_q
        return -log_q_sum / self.args.S
