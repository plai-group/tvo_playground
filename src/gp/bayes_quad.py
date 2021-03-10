import numpy as np
from src import util
import logging
import torch
from src.gp.BOv import BayesOpt
import matplotlib.pyplot as plt
from src import ml_helpers as mlh
from src.gp.BOv import unique_rows
import pickle

LEGEND_SIZE = 15
FIGURE_SIZE = (12, 8)
WINDOW = 5
K_MAX = 100
MIN_REL_CHANGE = 1e-3
MIN_ERR = 1e-3
LBM_GRID = np.linspace(-9, -0.1, 50)

emu_log = logging.getLogger("emukit")
emu_log.setLevel(logging.WARNING)

emu_gp = logging.getLogger("GP")
emu_gp.setLevel(logging.WARNING)


def extract_X_Y_from_args(SearchSpace,args,T=None):
    # obtain X and Y by truncating the data in the time-varying setting
    # if the existing data is <3*arg.K, randomly generate X

    if T is None:
        T=args.truncation_threshold
    lenY=len(args.Y_ori)
    X=args.X_ori[max(0,lenY-T):,1:-1] # remove the first and last  column which is 0 and 1

    ur=unique_rows(X)
    if sum(ur)<(3*args.K): # random search to get initial data
        init_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
        init_X=np.around(init_X, decimals=4)
        init_X=np.append(0,init_X)
        init_X=np.append(init_X,1)

        return np.sort(init_X),None

    if lenY%20==0:
        strPath="{:s}/save_X_Y_{:s}_S{:d}_K{:d}.p".format(str(args.artifact_dir), args.schedule, args.S, args.K)
        pickle.dump( [args.X_ori,args.Y_ori], open( strPath, "wb" ) )

    Y=np.reshape(args.Y_ori[max(0,lenY-T):],(-1,1))

    return X,Y

def append_Xori_Yori_from_args(args):
    # append the average logpx into Y
    # append the args.partition into X

    if args.len_terminated_epoch >0: # if we terminate the betas due to drip the epoch len is shorted
        average_y=np.mean(args.logtvopx_all[-args.len_terminated_epoch:])
    else:
        average_y=np.mean(args.logtvopx_all[-args.schedule_update_frequency:])
    args.average_y=np.append(args.average_y,average_y) # averaging the logpx over this window

    # error will happen at the first iteration when we add the first average_y into our data
    # this error is intentional, i will modify it by using a flag to indicate the first time
    if len(args.average_y)==1:
        print("ignore for the first time to save the first value of Y")
        return

    prev_y=args.average_y[-1] -args.average_y[-2]

    args.X_ori=np.vstack(( args.X_ori, np.reshape(format_input(args.partition),(1,args.K+1) )))
    args.Y_ori=np.append(args.Y_ori, prev_y)
    prev_X=np.round(args.X_ori[-1],decimals=4)
    print("X",prev_X,"Y",args.Y_ori[-1])



def calculate_BO_points(model,args):

    # process the input X and output Y from args
    append_Xori_Yori_from_args(args)

    SearchSpace=np.asarray([args.bandit_beta_min,args.bandit_beta_max]*(args.K-1)).astype(float) # this is the search range of beta from 0-1
    SearchSpace=np.reshape(SearchSpace,(args.K-1,2))

    if args.K==2:
        SearchSpace[0,1]=0.7
    else:
        ll=np.linspace(0,args.bandit_beta_max,args.K) # to discourage selecting 1
        for kk in range(args.K-1):
            SearchSpace[kk,1]=ll[kk+1]

    # truncate the time-varying data if neccessary
    # if dont have enough data -> randomly generate data
    if args.schedule=="gp": # non timevarying
        X,Y=extract_X_Y_from_args(SearchSpace,args,T=len(args.Y_ori))
    else:   # time varying
        X,Y=extract_X_Y_from_args(SearchSpace,args)

    if Y is None:
        return X

    # augment the data with artificial observations all zeros and all ones
    x_all_zeros=np.reshape(np.asarray([args.bandit_beta_min]*(args.K-1)),(1,-1))
    x_all_ones=np.reshape(np.asarray([args.bandit_beta_max]*(args.K-1)),(1,-1))

    worse_score=np.min(Y)

    X=np.vstack((X,x_all_zeros))
    X=np.vstack((X,x_all_ones))

    Y=np.vstack((Y,np.asarray(worse_score)))
    Y=np.vstack((Y,np.asarray(worse_score)))


    # perform GP bandit
    if args.schedule=="gp_bandit":
        myBO=BayesOpt(func=None,SearchSpace=SearchSpace)
    elif args.schedule=="tvgp" or args.schedule=="gp": # TV but not permutation invariant
        myBO=BayesOpt(func=None,SearchSpace=SearchSpace,GPtype="vanillaGP")
    else:
        print("please change ",args.schedule)


    myBO.init_with_data(X,Y)

    # beta is selected from here
    new_X=myBO.select_next_point()[1]

    # sorting
    new_X=np.round(new_X,decimals=4)
    new_X = np.append(np.append(0,np.sort(new_X)), 1)
    print(new_X)

    temp_new_X=np.unique(new_X)

    if np.array_equal(temp_new_X, [0, args.bandit_beta_min, 1]) or \
        np.array_equal(temp_new_X, [0, 1]) :#0.01 is due to the search bound
        rand_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
        return np.append(np.append(0,np.sort(rand_X)), 1)
    else:
        return new_X


def calculate_BO_points_vanillaGP(model,args): # this is used as a baseline with vanilla GP

    # process the input X and output Y from args
    append_Xori_Yori_from_args(args)

    SearchSpace=np.asarray([args.bandit_beta_min,args.bandit_beta_max]*(args.K-1)).astype(float) # this is the search range of beta from 0-1
    SearchSpace=np.reshape(SearchSpace,(args.K-1,2))

    if args.K==2:
        SearchSpace[0,1]=0.7
    else:
        ll=np.linspace(0,args.bandit_beta_max,args.K) # to discourage selecting 1
        for kk in range(args.K-1):
            SearchSpace[kk,1]=ll[kk+1]

    X,Y=extract_X_Y_from_args(SearchSpace,args,T=len(args.Y_ori)) # this is non timevarying GP, takes all data
    if Y is None:
        return X

    myBO=BayesOpt(func=None,SearchSpace=SearchSpace,GPtype="vanillaGP")
    myBO.init_with_data(X,Y)

    # beta is selected from here
    new_X=myBO.select_next_point()[1]
    new_X=np.round(new_X,decimals=4)

    new_X = np.append(np.append(0,np.sort(new_X)), 1)
    print(new_X)

    temp_new_X=np.unique(new_X)

    if np.array_equal(temp_new_X, [0, args.bandit_beta_min, 1]) or \
        np.array_equal(temp_new_X, [0, 1]) :#0.01 is due to the search bound
        rand_X = np.random.uniform(SearchSpace[:, 0], SearchSpace[:, 1],size=(1,args.K-1))
        return np.append(np.append(0,np.sort(rand_X)), 1)
    else:
        return new_X



def format_input(*data):
    output = []
    for d in data:
        if torch.is_tensor(d):
            d = d.cpu().numpy()
        if d.ndim == 1:
            d = np.expand_dims(d, 1)
        output.append(d)

    return output[0] if len(output) == 1 else output


def plot_variance(ivr_acquisition):
    x_plot = np.linspace(0, 1, 300)[:, None]
    ivr_plot = ivr_acquisition.evaluate(x_plot)

    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(x_plot, (ivr_plot - np.min(ivr_plot)) / (np.max(ivr_plot) - np.min(ivr_plot)),
             "green", label="integral variance reduction")

    plt.legend(loc=0, prop={'size': LEGEND_SIZE})
    plt.xlabel(r"$x$")
    plt.ylabel(r"$acquisition(x)$")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1.04)
    plt.show()


def get_cov_function(model, args):

    try: # for BDQ we already have log_weight, dont need to extract it again
        log_weight=args.log_weight
    except: # for other approaches, we need to extract it
        log_weight = util.get_total_log_weight(model, args, args.valid_S)

    def f(X):
        der = util.calc_var_given_betas(log_weight, args, all_sample_mean=True,
                                        betas = X)
        return der
    return f
