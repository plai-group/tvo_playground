# tvo_playground
Clean slate to begin research on the thermodynamic variational objective. The heart of the code is `src/models/base.py`, which is where most of the training logic lives. All new models should extend it to get access to the various losses. The outer training loop which calls all the code is `main.py`. 

It uses [sacred](https://sacred.readthedocs.io/en/stable/) for its command line interface and [wandb](https://www.wandb.com/) for experiment tracking. All hyperparameters are defined in `my_config()` and are available on the commandline via:

`python main.py with model_name='continuous_vae' loss='tvo' lr=0.001 --unobserved`

To save to your wandb database, drop the `--unobserved`:

`python main.py with model_name='continuous_vae' loss='tvo' lr=0.001`

Only strings, floats, ints, lists, and dicts should be defined in `my_config()`, everything else (numpy arrays, pickles, etc) should be instantiated in `init()`.


### losses
- tvo
- tvo_reparam
- reinforce
- elbo
- iwae
- iwae dreg
- vimco
- wake-wake (ww)
- wake-sleep (ws)

### datasets
- mnist
- fashion mnist
- kuzushiji mnist
- omniglot
- binarized mnist
- binarized omniglot
- pcfg

### schedules
- log
- linear
- moments
- gp_bandits

### models
- continuous vae
- discrete vaes
- bayesian neural networks (bnn)
- probabilistic context free grammar (pcfg)
