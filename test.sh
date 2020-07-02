#!/bin/bash
set -e
set -o pipefail

function print_and_run()
{
    echo "================"
    echo "Running $1"
    echo "================"
    eval $1
}

# small S, large batch, small epochs so test are fast
fast_settings='valid_S=5 test_S=5 batch_size=1000 test_batch_size=1000 epochs=2 record=False -p --unobserved'

# test all models
for model in continuous_vae discrete_vae bnn
do
    print_and_run "python main.py with model_name=$model loss='tvo' $fast_settings"
done

# test all models (pcfg faster w/ small batch, not vectorized)
for model in pcfg
do
    print_and_run "python main.py with model_name=$model loss='tvo' valid_S=5 test_S=5 epochs=2 record=False -p --unobserved"
done

# test all losses
for loss in elbo iwae_dreg iwae reinforce tvo_reparam tvo vimco wake-sleep wake-wake
do
    print_and_run "python main.py with model_name='continuous_vae' loss=$loss $fast_settings"
done


# test all schedules
for schedule in log linear moments
do
    print_and_run "python main.py with model_name='continuous_vae' loss='tvo' schedule=$schedule $fast_settings"
    print_and_run "python main.py with model_name='continuous_vae' loss='tvo_reparam' schedule=$schedule $fast_settings"
    echo
done

