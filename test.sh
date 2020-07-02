#!/bin/bash

function print_and_run()
{
    echo "================"
    echo "Running $1"
    echo "================"
    eval $1
}


print_and_run "python main.py with model_name='continuous_vae' loss='elbo' dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='iwae_dreg'  dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='iwae'  dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='reinforce' dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='tvo_reparam'  dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='tvo'  dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='vimco' dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='wake-sleep'  dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
print_and_run "python main.py with model_name='continuous_vae' loss='wake-wake'  dataset='mnist' valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"



# # check tvo
# for DATASET in mnist omniglot
# do
#     for SCHEDULE in moments
#     # for SCHEDULE in log linear moments coarse_grain
#     do
#         # check tvo
#         print_and_run "python main.py with model_name='continuous_vae' loss='tvo' dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"

#         # check tvo_reparam
#         print_and_run "python main.py with model_name='continuous_vae' loss='tvo_reparam' dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"

#         # check tvo w/ different K
#         print_and_run "python main.py with model_name='continuous_vae' K=10 loss='tvo'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
#         print_and_run "python main.py with model_name='continuous_vae' K=10 loss='tvo_reparam'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"

#         # check tvo w/ different S
#         print_and_run "python main.py with model_name='continuous_vae' S=10 loss='tvo'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
#         print_and_run "python main.py with model_name='continuous_vae' S=10 loss='tvo_reparam'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"

#         # check record
#         print_and_run "python main.py with model_name='continuous_vae' loss='tvo'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=True -p --unobserved"
#         print_and_run "python main.py with model_name='continuous_vae' loss='tvo_reparam'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=True -p --unobserved"
#     done
#     # check baselines
#     print_and_run "python main.py with model_name='continuous_vae' loss='iwae'       dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
#     print_and_run "python main.py with model_name='continuous_vae' loss='iwae_dreg'  dataset=$DATASET schedule=$SCHEDULE valid_S=5 test_S=5 test_batch_size=1000 epochs=2 record=False -p --unobserved"
#     echo $DATASET $SCHEDULE
# done
