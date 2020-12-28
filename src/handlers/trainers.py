from ..utils import losses
import torch


def train_epoch_dual_objective(model, data_loader, optimizer = None, args = None, record=False):
    optimizer = optimizer if optimizer is None else optimizer

    train_logpx = 0
    train_elbo = 0
    train_loss = 0

    for idx, data in enumerate(data_loader):
        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()
    
        loss = model.forward(data, set_internals = True)
        
        if args.loss == 'tvo_reparam': # p optimized using tvo
            theta_loss = losses.get_tvo_loss(internals)
        elif args.loss == 'tvo_q_only':
            # elbo calculated for final sample in flow chain
            theta_loss = internals.elbo
        #elif args.loss == 'iwae_dreg': # p optimized using IWAE (DReG update is only for q)
        #    theta_loss = losses.get_iwae_loss(interals)
        else:
            raise ValueError(
                "{} is an invalid loss".format(args.loss))
        wake_theta_loss.backward()
        optimizer_theta.step()

        optimizer_phi.zero_grad()
        optimizer_theta.zero_grad()


        if args.loss in ['tvo_reparam', 'tvo_q_only']:
            sleep_phi_loss = losses.get_tvo_loss(internals)
            ##sleep_phi_loss = losses.get_tvo_reparam_loss(data)
            sleep_phi_loss.backward()
        #elif args.loss == 'iwae_dreg':
        #    sleep_phi_loss = losses.get_iwae_dreg_loss(data)
        #    sleep_phi_loss.backward()
        else:
            raise ValueError(
                "{} is an invalid loss".format(args.loss))
        optimizer_phi.step()


        # if record: #args.record:
        #     self.record_stats()

        iwae, elbo = losses.evaluate_lower_bounds(model, data, args.valid_chains)

        train_loss += loss
        train_logpx += iwae.item()
        train_elbo += elbo.item()

    train_loss = train_loss / len(data_loader)
    train_logpx = train_logpx / len(data_loader)
    train_elbo = train_elbo / len(data_loader)

    #if record: self.save_record() #args.record: self.save_record()
    #self.last_training_batch = data

    return train_loss, train_logpx, train_elbo


def get_loss(elbo, args):
    p_loss = None
    if args.loss == 'iwae':
        q_loss = losses.iwae_loss(elbo, num_chains = num_chains, dim=1)
    elif args.loss == 'elbo':
        q_loss = elbo
    elif args.loss == 'tvo':
        print("TO DO : calc TVO loss ")
        import IPython
        IPython.embed()
    elif args.loss == 'multi_iwae':
        pass

    if p_loss is None:
        return -torch.mean(q_loss)
    else:
        return -torch.mean(q_loss), -torch.mean(p_loss) if p_loss is not None else p_loss


def train_epoch_single_objective(model, data_loader, optimizer, args, record=False, itr=None):
    train_logpx = 0
    train_loss = 0
    train_elbo = 0 
    for idx, data in enumerate(data_loader):
        data = data[0] if isinstance(data, list) else data
        optimizer.zero_grad()


        #loss = model.forward(data, itr=itr)
        elbo = model.forward(data, itr=itr)

        loss = get_loss(elbo, args)


        loss.backward()
        optimizer.step()

        iwae, elbo = losses.evaluate_lower_bounds(model, data, args.valid_chains)

        #test_elbo = self.get_test_elbo(data, self.args.valid_S)

        #if record:  # self.args.record:
        #    self.record_stats()

        train_loss += loss

        train_logpx += iwae.item()
        train_elbo += elbo.item()

    train_loss = train_loss / len(data_loader)
    train_logpx = train_logpx / len(data_loader)
    train_elbo = train_elbo / len(data_loader)

    # if record:
    #     self.save_record()

    return train_loss, train_logpx, train_elbo


def test_eval(model, data_loader, args):
    test_logpx = 0
    test_loss = 0
    test_elbo = 0 
    for idx, data in enumerate(data_loader):
        data = data[0] if isinstance(data, list) else data
        optimizer.zero_grad()

        with torch.no_grad():
            #loss = model.forward(data, itr=itr)
            elbo = model.forward(data, itr=itr)

            loss = get_loss(elbo, args)


            # loss.backward()
            # optimizer.step()

            iwae, elbo = losses.evaluate_lower_bounds(model, data, args.test_chains)

            #test_elbo = self.get_test_elbo(data, self.args.valid_S)

            #if record:  # self.args.record:
            #    self.record_stats()

            test_elbo += loss

            test_logpx += iwae.item()
            train_elbo += elbo.item()

    test_loss = test_loss / len(data_loader)
    test_elbo = test_logpx / len(data_loader)
    train_elbo = train_elbo / len(data_loader)

    # if record:
    #     self.save_record()

    return train_loss, test_logpx, train_elbo