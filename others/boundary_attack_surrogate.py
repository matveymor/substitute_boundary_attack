from numpy.linalg import norm
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import sys
import random
from Boundary_attack import forward_perturbation, boundary_attack, get_diff

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1   = nn.Linear(576, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



#функция проекции объекта на адверсальную область вокруг исходного изображения
def projection(net, init_image, target_image):
    epsilon = 1
    predict_init = net(torch.tensor(init_image.reshape(1, 3, 32, 32)).to('cuda:0')).cpu().data.numpy()
    attack_class = np.argmax(predict_init)
    
    while True:
        samples = []
        for i in range(50):
            trial_sample = init_image + forward_perturbation(epsilon * get_diff(init_image, target_image), 
                                                                 init_image.reshape(3, 32, 32), target_image.reshape(3, 32, 32)).reshape(3,32,32)
            samples.append(trial_sample)
            epsilon *= 0.95
        
        samples = np.array(samples)
        prediction = net(torch.tensor(samples.reshape(50, 3, 32, 32)).to('cuda:0')).cpu().data.numpy()
        prediction = np.argmax(prediction, axis=1)
        indexes = np.where(prediction == attack_class)[0]
        
        if len(indexes) > 0:
            proj_sample = samples[indexes[0]]
            return proj_sample
            


def get_projection_class(net, train_data, train_targets, target_image, label, len_class, step):
    label = torch.tensor(label)
    class_2 = []
    print('Getting projection set...')
    print('Size of the set:', len_class+step)
    train_data = train_data[train_targets!=label]
    train_targets = train_targets[train_targets!=label]
    
    for i in range(step, len_class+step):
        if i >= 9000:
            break
            
        if train_targets[i] != label:
            sys.stdout.write("\rImage number: %d" % (i,))
            init_image = train_data[i].data.numpy().reshape(3, 32, 32)
            proj_sample = projection(net, init_image, target_image)
            class_2.append(proj_sample)
        else:
            step += 1
            
    print('')
    return class_2


def train_epoch(model, optimizer, loss, X_train, y_train, X_test, y_test):
    running_loss = 0.0
    train_acc = 0.0
    test_acc = 0.0
    len_train = 0.0
    len_test = 0.0
    loss_train = 0.0

    batch_size = 32

    # new_index = np.random.permutation(len(X_train_))
    # X_train = X_train_[new_index]
    # y_train = y_train_[new_index]

    for step in range(0, len(X_train), batch_size):
        inputs, targets = X_train[step:step + batch_size], y_train[step:step + batch_size],
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_val = loss(outputs, targets)
        loss_val.backward()
        optimizer.step()
        accuracy_add = (outputs.argmax(dim=1) == targets).float().sum().data.cpu()
        train_acc += accuracy_add
        len_train += len(targets)
        loss_train += len(targets) * loss_val.item()
        running_loss += loss_val.item()

    inputs, targets = X_test, y_test
    outputs = model(inputs)
    #outputs = torch.nn.functional.softmax(outputs)
    #loss_val = loss(outputs, targets)
    accuracy_add = (outputs.argmax(dim=1) == targets).float().sum().data.cpu()
    test_acc += accuracy_add
    len_test += len(targets)

    return model, train_acc / len_train, test_acc / len_test


def train(model, optimizer, loss, X_train, y_train, X_test, y_test):
    accuracy_history_test = []
    accuracy_history_train = []

    for epoch in range(0, 20):
        model, acc_train, acc_test = train_epoch(model, optimizer, loss, X_train, y_train, X_test, y_test)
        accuracy_history_test.append(acc_train)
        accuracy_history_train.append(acc_test)
        print('Epoch:', epoch, '   acc_train:', acc_train, '   acc_test:', acc_test)
        if acc_test > 0.96 and epoch > 3:
            break
            
    return model


def get_surrogate_model(X, y, model_name='alexnet', model=None):
    # В качестве лоса возмем кросс-энтропию. Оптимизатор - Адам
    print('')
    print('Train surrogate model')
    loss = torch.nn.CrossEntropyLoss()

    len_set = int(len(X)*0.92)
    X_train = X[:len_set]
    X_test = X[len_set:]
    y_train = y[:len_set]
    y_test = y[len_set:]
    
    if model is None:
        if model_name == 'alexnet':
            model_surrogate = torchvision.models.alexnet(pretrained=False)
        elif model_name == 'lenet':
            model_surrogate = LeNet()
        else:
            raise Exception('You can not choose such model!')
    else:
        model_surrogate = model
        
    model_surrogate = model_surrogate.to('cuda:0')        
    optimizer = torch.optim.Adam(model_surrogate.parameters(), lr=1.0e-3)
    model_surrogate = train(model_surrogate, optimizer, loss, X_train, y_train, X_test, y_test)   
    model_surrogate = model_surrogate.to('cpu')
    model_surrogate = model_surrogate.eval()
    print('\n')
    return model_surrogate


def prepare_dataset(class_1, class_2):
    X = np.concatenate((class_1, class_2), axis=0)
    y_1 = np.zeros(len(class_1))
    y_2 = np.ones(len(class_2))
    y = np.append(y_1, y_2)
    y = torch.tensor(y, dtype=torch.long)
    X = torch.tensor(X, dtype=torch.float).reshape(len(X), 3, 32, 32)

    y = y.to('cuda:0')
    X = X.to('cuda:0')

    new_index = np.random.permutation(len(X))

    y = y[new_index]
    X = X[new_index]
    return X, y



def prepare_data_grad(model, example, example_target):
    loss = torch.nn.CrossEntropyLoss()
    output = model(example)
    loss_val = loss(output, example_target.reshape(-1))
    model.zero_grad() 
    loss_val.backward()
    data_grad = example.grad.data
    return data_grad



def gradient_step_surrogate(model_surrogate, net, label, image, target):
    print('Start gradient attack')
    image = torch.tensor(image, dtype=torch.float).reshape(1,1,28,28)
    image_init = image
    image.requires_grad = True
    #target = torch.tensor(1).reshape(-1)
    epsilon = 0.01
    i = 0
    loss = torch.nn.CrossEntropyLoss()
    
    while True:
        data_grad = prepare_data_grad(model_surrogate, image, target)
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        adv_label = np.argmax(model_surrogate(perturbed_image).data.numpy())
        image = torch.tensor(perturbed_image, requires_grad=True)
        i += 1
        
        predict_init = net(torch.tensor(image.reshape(1, 1, 28, 28))).to('cpu').data.numpy()
        attack_class = np.argmax(predict_init)

        if attack_class != label:
            print('Adversarial label:', attack_class, '  Iteration num:', i)
            break
            
    image.requires_grad = False
    image = image.data.numpy().reshape(28,28)
            
    return image
    
    

#фукнция для поиска ближайшего изображения в другом классе
def nearest_image(image, label, label_init, train_data, train_targets):
    all_norm = []
    for i, train_image in enumerate(train_data):
        if (train_targets[i].data.cpu().numpy() != label):
            train_image = train_image.data.cpu().numpy()
            diff = train_image - image
            norm_diff = norm(diff)
            all_norm.append(norm_diff)
        else:
            all_norm.append(1e20)
            
    all_norm = np.array(all_norm)
    index = np.argmin(all_norm)
    adv_label = train_targets[index].data.numpy()
    train_targets = train_targets.data.numpy()
    all_norm[np.where(train_targets != adv_label)[0]] = 1e20
    
    indexes = []
    for i in range(10):
        index = np.argmin(all_norm)
        indexes.append(index)
        all_norm[index] = 1e20
        
        
    print(train_targets[indexes])
        
    return indexes


def our_attack(net, target_image, label, test_data, train_data, train_targets, model_name='alexnet', verbose=1, threshold=None, max_iter=None, mu=0.0):
    print('Init label:', label)
    print('')
    target_image = target_image.data.numpy().reshape(3, 32, 32)

    indexes = nearest_image(target_image, label, 11, train_data, train_targets)
    class_1 = []
    initiation = []
    initiation_index = []
    
    for index in indexes:
        x_init = train_data[index].data.numpy().reshape(3, 32, 32)
        print('Adversarial label:', train_targets[index].data.numpy())       
        class_1_, adversarial_last, distance_init = boundary_attack(net, x_init, target_image, threshold=None, verbose=1, max_iter=5e2, mu=0.0)
        
        norm_ = np.linalg.norm(adversarial_last - target_image)
        if norm_ > 1e-8:
            initiation.append(adversarial_last)
            initiation_index.append(norm_)
        else:
            initiation_index.append(1e20)
            
        print('')
        if len(class_1) == 0:
            class_1 = class_1_
        else:
            if len(class_1_) > 0:
                class_1 = np.concatenate((class_1, class_1_), axis=0)
                
    index = indexes[np.argmin(initiation_index)]
        
    net = net.to('cpu')
    predict_adv = net(torch.tensor(adversarial_last.reshape(1, 3, 32, 32), dtype=torch.float)).to('cpu').data.numpy()
    adv_class = np.argmax(predict_adv)
    print('Adversarial label', adv_class)
    net = net.to('cuda:0')
    class_2 = get_projection_class(net, train_data, train_targets, target_image, label, len(class_1), 0)

    X, y = prepare_dataset(class_1, class_2)
    
    model_surrogate = get_surrogate_model(X, y, model_name=model_name)
    #x_init = gradient_step_surrogate(model_surrogate, net, label, target_image, torch.tensor(0).reshape(-1))
    net = net.to('cuda:0')
    x_init = train_data[index].data.numpy().reshape(3, 32, 32)
    adversarials_wrong__, x_adv_init, distance_init = boundary_attack(net, x_init, target_image, threshold=None, verbose=1, max_iter=11e3, mu=0, surrogate_model=None)
    adversarials_wrong, x_init, distance_our = boundary_attack(net, x_init, target_image, threshold=None, verbose=1, max_iter=1000, mu=1e-4, surrogate_model=model_surrogate)
    x_init = np.array(x_init, dtype= np.float64)
    print('')
    k = 0
    mu = 1e-4
    while True:

        if k > 10: break
        step = len(class_1)
        print('Attack', k)
        if len(adversarials_wrong) > 0:
            class_1 = np.concatenate((class_1, adversarials_wrong), axis=0)
        print('')

        if len(adversarials_wrong) > 0:
            class_2_ = get_projection_class(net, train_data, train_targets, target_image, label, len(adversarials_wrong), step)
            if len(class_2_) > 0:
                class_2 = np.concatenate((class_2, class_2_), axis=0)
            X, y = prepare_dataset(class_1, class_2)
            
        if len(adversarials_wrong) == 0:
            mu = 0.7 * mu
            print('New momentum:', mu)
            mu = np.maximum(mu, 3e-5)
        
        model_surrogate = get_surrogate_model(X, y, model_name=model_name, model=model_surrogate)
        #x_init = gradient_step_surrogate(model_surrogate, net, label, target_image, torch.tensor(1).reshape(-1))
        net = net.to('cuda:0')
        #x_init = train_data[index].data.numpy().reshape(28, 28)

        adversarials_wrong, x_init, distance = boundary_attack(net, x_init, target_image, threshold=None, verbose=1, max_iter=1000, mu=mu, surrogate_model=model_surrogate)
        x_init = np.array(x_init, dtype=np.float64)
        distance_our = np.append(distance_our, distance)
        k += 1
        
    return x_init, distance_our, distance_init















