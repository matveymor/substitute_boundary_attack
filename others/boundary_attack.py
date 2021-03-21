import numpy as np
import torch
import sys


def get_diff(sample_1, sample_2):
    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i])))
        
    diff = np.array(diff)
    return np.array(diff)


def orthogonal_perturbation(delta, prev_sample, target_sample):
    prev_sample = prev_sample.reshape(3, 32, 32)

    perturb = torch.tensor(np.random.randn(3, 32, 32))
    get_diff_ = torch.tensor(get_diff(torch.tensor(perturb), torch.tensor(np.zeros_like(perturb))))
    perturb /= get_diff_.reshape(3, 1, 1)
    perturb *= delta * np.mean(get_diff(target_sample, prev_sample))

    diff = (target_sample - prev_sample)
    diff /= torch.tensor(get_diff(target_sample, prev_sample)).reshape(3, 1, 1)
    diff = diff.reshape(3, 32, 32)
    perturb = perturb.reshape(3, 32, 32).to(dtype=torch.float)
    for i, channel in enumerate(diff):
        difference = torch.tensor(np.dot(perturb[i].to('cpu', dtype=torch.float).detach().numpy(), 
                               channel.to('cpu', dtype=torch.float).detach().numpy()) 
                                * channel.to('cpu', dtype=torch.float).detach().numpy())
        perturb[i] -= difference
    
    perturb = perturb.data.numpy()
    mean = target_sample

    mean = np.array([np.mean(mean[0]), np.mean(mean[1]), np.mean(mean[2])])
    
    perturb = perturb.reshape(3, 32, 32)
    
    overflow = (prev_sample + perturb) - np.concatenate((np.ones((1, 32, 32)) * (255. - mean[0]), 
                                                                      np.ones((1, 32, 32)) * (255. - mean[1]), 
                                                                      np.ones((1, 32, 32)) * (255. - mean[2])), axis=0)
    overflow = overflow.reshape(3, 32, 32)
    overflow = np.nan_to_num(overflow)
    perturb -= overflow * (overflow > 0.)
    
    underflow = -(prev_sample + perturb) + np.concatenate((np.ones((1, 32, 32)) * (0. - mean[0]), 
                                             np.ones((1, 32, 32)) * (0. - mean[1]), 
                                             np.ones((1, 32, 32)) * (0. - mean[2])), axis=0)
               
    underflow = underflow.reshape(3, 32, 32)
    underflow = np.nan_to_num(underflow)
    perturb += underflow * (underflow > 0)
    perturb = np.nan_to_num(perturb)
    return perturb

def forward_perturbation(epsilon, prev_sample, target_sample):
    target_sample = torch.tensor(target_sample)
    prev_sample = torch.tensor(prev_sample)
    perturb = (target_sample - prev_sample)
    perturb = perturb.reshape(32, 32, 3)
    perturb /= torch.tensor(get_diff(target_sample, prev_sample))
    perturb *= torch.tensor(epsilon)
    perturb = perturb.data.numpy()
    perturb = np.nan_to_num(perturb)
    return perturb


def prepare_data_grad(model, example, example_target):
    loss = torch.nn.CrossEntropyLoss()
    output = model(example)
    loss_val = loss(output, example_target.reshape(-1))
    model.zero_grad()
    loss_val.backward()
    data_grad = example.grad.data
    return data_grad


def boundary_attack(net, initial_image, target_image, verbose=1, threshold=None, max_iter=None, mu=3e-4, surrogate_model=None):
    if verbose > 0:
        print('Start boundary attack')

    predict_init = net(torch.tensor(initial_image.reshape(1, 3, 32, 32), dtype=torch.float).to('cuda:0')).to('cpu').data.numpy()
    predict_target = net(torch.tensor(target_image.reshape(1, 3, 32, 32)).to('cuda:0')).to('cpu').data.numpy()
    attack_class = np.argmax(predict_init)

    adversarial_sample = initial_image
    adversarial_momentum = None

    n_steps = 0
    n_calls = 0
    epsilon = 1.
    delta = 0.1
    label_surrogate = 0
    
    distances = []

    adversarials_wrong = []

    # Шаг 1. Находим проекцию на границу адверсальности
    while True:
        trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_image), 
                                                                 adversarial_sample.reshape(3, 32, 32), target_image.reshape(3, 32, 32)).reshape(3,32,32)
        prediction = net(torch.tensor(trial_sample.reshape(1, 3, 32, 32), dtype=torch.float).to('cuda:0')).to('cpu').data.numpy()
        n_calls += 1
        if np.argmax(prediction) == attack_class:
            adversarial_last = trial_sample
            distance_last = np.linalg.norm(adversarial_last - target_image)
            adversarial_sample = trial_sample
            break
        else:
            epsilon *= 0.9

    # Шаг 2. Дельта шаг
    eps_step = epsilon
    delta_step = delta
    while True:
        d_step = 0
        delta_init = delta
        while True:
            d_step += 1
            trial_samples = []
            ort_perts = []
            if not adversarial_momentum is None:
                trial_sample_init = adversarial_last + mu * adversarial_momentum
            else:
                trial_sample_init = adversarial_last
                
            for _ in np.arange(10):
                ort_pert = orthogonal_perturbation(delta, trial_sample_init, target_image)
                trial_sample = trial_sample_init + (1.-mu)*ort_pert

                trial_samples.append(trial_sample)
                ort_perts.append(ort_pert)

            trial_samples = np.array(trial_samples)
            predictions = net(torch.tensor(trial_samples.reshape(10, 3, 32, 32)).to(dtype=torch.float).to('cuda:0')).to('cpu').data.numpy()
            n_calls += 10
            predictions = np.argmax(predictions, axis=1)
            d_score = np.mean(predictions == attack_class)
            if d_score > 0.0:
                if d_score < 0.3:
                    delta *= 0.9
                elif d_score > 0.7:
                    delta /= 0.9
                adversarial_sample = np.array(trial_samples)[np.where(predictions == attack_class)[0][0]]

                if d_score < 1:
                    adversarial_wrong = np.array(trial_samples)[np.where(predictions != attack_class)[0][0]]
                else:
                    adversarial_wrong = None

                ort_pert = np.array(ort_perts)[np.where(predictions == attack_class)[0][0]]

                pred_momentum = np.argmax(net(torch.tensor(adversarial_sample.reshape(1, 3, 32, 32)).to(dtype=torch.float).to('cuda:0')).to('cpu').data.numpy(), axis=1)

                if pred_momentum == attack_class:
                    break
            else:
                delta *= 0.95

            if delta < 1e-10:
                delta_init *= 2
                delta += delta_init

        e_step = 0
        while True:
            e_step += 1
            trial_sample = adversarial_sample + forward_perturbation(epsilon * get_diff(adversarial_sample, target_image), 
                                                                 adversarial_sample.reshape(3, 32, 32), target_image.reshape(3, 32, 32)).reshape(3,32,32)
            prediction = net(torch.tensor(trial_sample.reshape(1, 3, 32, 32)).to('cuda:0', dtype=torch.float)).to('cpu').data.numpy()
            n_calls += 1
            if np.argmax(prediction) == attack_class:
                adversarial_sample = trial_sample
                epsilon /= 0.5
                break
            elif e_step > 500:
                break
            else:
                epsilon *= 0.5
        n_steps += 1

        distance = np.linalg.norm(adversarial_sample - target_image)

        if distance > distance_last:
            adversarial_sample = adversarial_last
            epsilon = eps_step
            delta = delta_step
        else:
            if not surrogate_model is None:
                adv_grad = torch.tensor(adversarial_sample).reshape(1, 3, 32, 32).to(dtype=torch.float)
                label_surrogate = surrogate_model(adv_grad).data.numpy()
                label_surrogate = np.argmax(label_surrogate)
                adv_grad.requires_grad = True
                adversarial_momentum = prepare_data_grad(surrogate_model, adv_grad, torch.tensor(0).reshape(-1))
                #adversarial_momentum = adversarial_momentum.sign()
                adversarial_momentum = adversarial_momentum.data.numpy().reshape(3, 32, 32)
                adversarial_momentum /= np.linalg.norm(adversarial_momentum)
            adversarial_last = adversarial_sample
            distance_last = distance
            delta_step = delta
            eps_step = epsilon

            if not adversarial_wrong is None:
                adversarials_wrong.append(adversarial_wrong)
                '''
                if len(adversarials_wrong) == 0:
                    adversarials_wrong = adversarial_wrong
                else:
                    adversarials_wrong = np.concatenate((adversarials_wrong, adversarial_wrong), axis=0)
                '''
                
        distances.append(distance_last)

        if verbose > 0:
            sys.stdout.write("\rdistance: %0.4f, itetarion: %d, label_sur: %d, delta: %0.4f, eps: %0.5f" % (distance_last, n_steps, label_surrogate, delta_step, eps_step))

        if not threshold is None:
            if distance_last <= threshold:
                break

        if not max_iter is None:
            if n_steps > max_iter:
                break
    print('')
    return adversarials_wrong, adversarial_last, distances