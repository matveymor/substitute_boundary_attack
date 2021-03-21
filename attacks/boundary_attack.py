from typing import Union, Tuple, Optional, Any
from typing_extensions import Literal
import numpy as np
import eagerpy as ep
import logging

from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd

from foolbox.types import Bounds

from foolbox.models import Model

from foolbox.criteria import Criterion

from foolbox.distances import l2

from foolbox.tensorboard import TensorBoard

from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack

from foolbox.attacks.base import MinimizationAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import get_is_adversarial
from foolbox.attacks.base import raise_if_kwargs


####################
#My imports

import torch
import torchvision
import foolbox as fb
import eagerpy as ep

from tqdm import tqdm
from time import time
from typing import Union, Any, Optional, Callable, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from foolbox.models.base import Model

torch.manual_seed(42)


def train_epoch(model, optimizer, loss, X_train, y_train, X_test, y_test):
    running_loss = 0.0
    train_acc = 0.0
    test_acc = 0.0
    len_train = 0.0
    len_test = 0.0
    loss_train = 0.0

    batch_size = 32
    
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
    accuracy_add = (outputs.argmax(dim=1) == targets).float().sum().data.cpu()
    test_acc += accuracy_add
    len_test += len(targets)
    
    y_pred = outputs.argmax(dim=1).data.cpu()
    auc = roc_auc_score(targets.cpu(), y_pred)
    
    return model, train_acc / len_train, test_acc / len_test, auc



def train(model, optimizer, loss, X_train, y_train, X_test, y_test):
    accuracy_history_test = []
    accuracy_history_train = []
    
    verbose = False
    
    for epoch in range(0, 10):
        model, acc_train, acc_test, auc = train_epoch(model, optimizer, loss, X_train, y_train, X_test, y_test)
        accuracy_history_test.append(acc_test)
        accuracy_history_train.append(acc_train)
        
        if verbose:
            print('Epoch:', epoch+1, '  ||   acc_train:', np.round(acc_train.numpy(), 3), '  ||   acc_test:', np.round(acc_test.numpy(), 3), '  ||   auc_test:', np.round(auc, 3))
            
    return model, accuracy_history_test, accuracy_history_train




class BoundaryAttack(MinimizationAttack):
    """A powerful adversarial attack that requires neither gradients
    nor probabilities.

    This is the reference implementation for the attack. [#Bren18]_

    Notes:
        Differences to the original reference implementation:
        * We do not perform internal operations with float64
        * The samples within a batch can currently influence each other a bit
        * We don't perform the additional convergence confirmation
        * The success rate tracking changed a bit
        * Some other changes due to batching and merged loops

    Args:
        init_attack : Attack to use to find a starting points. Defaults to
            LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        steps : Maximum number of steps to run. Might converge and stop before that.
        spherical_step : Initial step size for the orthogonal (spherical) step.
        source_step : Initial step size for the step towards the target.
        source_step_convergance : Sets the threshold of the stop criterion:
            if source_step becomes smaller than this value during the attack,
            the attack has converged and will stop.
        step_adaptation : Factor by which the step sizes are multiplied or divided.
        tensorboard : The log directory for TensorBoard summaries. If False, TensorBoard
            summaries will be disabled (default). If None, the logdir will be
            runs/CURRENT_DATETIME_HOSTNAME.
        update_stats_every_k :

    References:
        .. [#Bren18] Wieland Brendel (*), Jonas Rauber (*), Matthias Bethge,
           "Decision-Based Adversarial Attacks: Reliable Attacks
           Against Black-Box Machine Learning Models",
           https://arxiv.org/abs/1712.04248
    """

    distance = l2

    def __init__(
        self,
        init_attack: Optional[MinimizationAttack] = None,
        steps: int = 25000,
        spherical_step: float = 1e-2,
        source_step: float = 1e-2,
        source_step_convergance: float = 1e-7,
        step_adaptation: float = 1.5,
        tensorboard: Union[Literal[False], None, str] = False,
        update_stats_every_k: int = 10,
    ):
        if init_attack is not None and not isinstance(init_attack, MinimizationAttack):
            raise NotImplementedError
        self.init_attack = init_attack
        self.steps = steps
        self.spherical_step = spherical_step
        self.source_step = source_step
        self.source_step_convergance = source_step_convergance
        self.step_adaptation = step_adaptation
        self.tensorboard = tensorboard
        self.update_stats_every_k = update_stats_every_k

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        originals, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if starting_points is None:
            init_attack: MinimizationAttack
            if self.init_attack is None:
                init_attack = LinearSearchBlendedUniformNoiseAttack(steps=50)
                logging.info(
                    f"Neither starting_points nor init_attack given. Falling"
                    f" back to {init_attack!r} for initialization."
                )
            else:
                init_attack = self.init_attack
            # TODO: use call and support all types of attacks (once early_stop is
            # possible in __call__)
            best_advs = init_attack.run(
                model, originals, criterion, early_stop=early_stop
            )
        else:
            best_advs = ep.astensor(starting_points)

        is_adv = is_adversarial(best_advs)
        if not is_adv.all():
            failed = is_adv.logical_not().float32().sum()
            if starting_points is None:
                raise ValueError(
                    f"init_attack failed for {failed} of {len(is_adv)} inputs"
                )
            else:
                raise ValueError(
                    f"{failed} of {len(is_adv)} starting_points are not adversarial"
                )
        del starting_points

        tb = TensorBoard(logdir=self.tensorboard)

        N = len(originals)
        ndim = originals.ndim
        spherical_steps = ep.ones(originals, N) * self.spherical_step
        source_steps = ep.ones(originals, N) * self.source_step

        tb.scalar("batchsize", N, 0)

        # create two queues for each sample to track success rates
        # (used to update the hyper parameters)
        stats_spherical_adversarial = ArrayQueue(maxlen=100, N=N)
        stats_step_adversarial = ArrayQueue(maxlen=30, N=N)

        bounds = model.bounds
        
        self.class_1 = []
        self.class_2 = []
        
        
        self.surrogate_model = None
        device = model.device
        train_step = 500

        for step in tqdm(range(1, self.steps + 1)):
            converged = source_steps < self.source_step_convergance
            if converged.all():
                break  # pragma: no cover
            converged = atleast_kd(converged, ndim)

            # TODO: performance: ignore those that have converged
            # (we could select the non-converged ones, but we currently
            # cannot easily invert this in the end using EagerPy)

            unnormalized_source_directions = originals - best_advs
            source_norms = ep.norms.l2(flatten(unnormalized_source_directions), axis=-1)
            source_directions = unnormalized_source_directions / atleast_kd(
                source_norms, ndim
            )

            # only check spherical candidates every k steps
            check_spherical_and_update_stats = step % self.update_stats_every_k == 0

            candidates, spherical_candidates = draw_proposals(
                bounds,
                originals,
                best_advs,
                unnormalized_source_directions,
                source_directions,
                source_norms,
                spherical_steps,
                source_steps,
                self.surrogate_model
            )
            candidates.dtype == originals.dtype
            spherical_candidates.dtype == spherical_candidates.dtype

            is_adv = is_adversarial(candidates)
            is_adv_spherical_candidates = is_adversarial(spherical_candidates)
            
            if is_adv.item():
                self.class_1.append(candidates)
            
            if not is_adv_spherical_candidates.item():
                self.class_2.append(spherical_candidates)
                
            
            if (step%train_step==0) and (step>0):
                
                start_time = time()
                
                class_1 = self.class_1
                class_2 = self.class_2

                class_1 = np.array([image.numpy()[0] for image in class_1])
                class_2 = np.array([image.numpy()[0] for image in class_2])

                class_2 = class_2[:len(class_1)]
                data = np.concatenate([class_1, class_2])
                labels = np.append(np.ones(len(class_1)), np.zeros(len(class_2)))
                
                X = torch.tensor(data).to(device)
                y = torch.tensor(labels, dtype=torch.long).to(device)
                
                if self.surrogate_model is None:
                    model_sur = torchvision.models.resnet18(pretrained=True)
                    #model.features[0] = torch.nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                    model_sur.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
                    model_sur = model_sur.to(device)
                else:
                    model_sur = model_surrogate
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                optimizer = torch.optim.Adam(model_sur.parameters(), lr=3e-4)
                loss = torch.nn.CrossEntropyLoss()
                
                model_surrogate, accuracy_history_test, accuracy_history_train = train(model_sur, optimizer, loss, X_train, y_train, X_test, y_test)
                model_surrogate = model_surrogate.eval()
                
                self.surrogate_model = fb.PyTorchModel(model_surrogate, bounds=(0, 1), device=device)
                
                end_time = time()
                
                #print('Time for train: ', np.round(end_time - start_time, 2))
                #print('\n')

            spherical_is_adv: Optional[ep.Tensor]
            if check_spherical_and_update_stats:
                spherical_is_adv = is_adversarial(spherical_candidates)
                stats_spherical_adversarial.append(spherical_is_adv)
                # TODO: algorithm: the original implementation ignores those samples
                # for which spherical is not adversarial and continues with the
                # next iteration -> we estimate different probabilities (conditional vs. unconditional)
                # TODO: thoughts: should we always track this because we compute it anyway
                stats_step_adversarial.append(is_adv)
            else:
                spherical_is_adv = None

            # in theory, we are closer per construction
            # but limited numerical precision might break this
            distances = ep.norms.l2(flatten(originals - candidates), axis=-1)
            closer = distances < source_norms
            is_best_adv = ep.logical_and(is_adv, closer)
            is_best_adv = atleast_kd(is_best_adv, ndim)

            cond = converged.logical_not().logical_and(is_best_adv)
            best_advs = ep.where(cond, candidates, best_advs)

            tb.probability("converged", converged, step)
            tb.scalar("updated_stats", check_spherical_and_update_stats, step)
            tb.histogram("norms", source_norms, step)
            tb.probability("is_adv", is_adv, step)
            if spherical_is_adv is not None:
                tb.probability("spherical_is_adv", spherical_is_adv, step)
            tb.histogram("candidates/distances", distances, step)
            tb.probability("candidates/closer", closer, step)
            tb.probability("candidates/is_best_adv", is_best_adv, step)
            tb.probability("new_best_adv_including_converged", is_best_adv, step)
            tb.probability("new_best_adv", cond, step)

            if check_spherical_and_update_stats:
                full = stats_spherical_adversarial.isfull()
                tb.probability("spherical_stats/full", full, step)
                if full.any():
                    probs = stats_spherical_adversarial.mean()
                    cond1 = ep.logical_and(probs > 0.5, full)
                    spherical_steps = ep.where(
                        cond1, spherical_steps * self.step_adaptation, spherical_steps
                    )
                    source_steps = ep.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = ep.logical_and(probs < 0.2, full)
                    spherical_steps = ep.where(
                        cond2, spherical_steps / self.step_adaptation, spherical_steps
                    )
                    source_steps = ep.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
                    tb.conditional_mean(
                        "spherical_stats/isfull/success_rate/mean", probs, full, step
                    )
                    tb.probability_ratio(
                        "spherical_stats/isfull/too_linear", cond1, full, step
                    )
                    tb.probability_ratio(
                        "spherical_stats/isfull/too_nonlinear", cond2, full, step
                    )

                full = stats_step_adversarial.isfull()
                tb.probability("step_stats/full", full, step)
                if full.any():
                    probs = stats_step_adversarial.mean()
                    # TODO: algorithm: changed the two values because we are currently tracking p(source_step_sucess)
                    # instead of p(source_step_success | spherical_step_sucess) that was tracked before
                    cond1 = ep.logical_and(probs > 0.25, full)
                    source_steps = ep.where(
                        cond1, source_steps * self.step_adaptation, source_steps
                    )
                    cond2 = ep.logical_and(probs < 0.1, full)
                    source_steps = ep.where(
                        cond2, source_steps / self.step_adaptation, source_steps
                    )
                    stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
                    tb.conditional_mean(
                        "step_stats/isfull/success_rate/mean", probs, full, step
                    )
                    tb.probability_ratio(
                        "step_stats/isfull/success_rate_too_high", cond1, full, step
                    )
                    tb.probability_ratio(
                        "step_stats/isfull/success_rate_too_low", cond2, full, step
                    )

            tb.histogram("spherical_step", spherical_steps, step)
            tb.histogram("source_step", source_steps, step)
        tb.close()
        return restore_type(best_advs)


class ArrayQueue:
    def __init__(self, maxlen: int, N: int):
        # we use NaN as an indicator for missing data
        self.data = np.full((maxlen, N), np.nan)
        self.next = 0
        # used to infer the correct framework because this class uses NumPy
        self.tensor: Optional[ep.Tensor] = None

    @property
    def maxlen(self) -> int:
        return int(self.data.shape[0])

    @property
    def N(self) -> int:
        return int(self.data.shape[1])

    def append(self, x: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = x
        x = x.numpy()
        assert x.shape == (self.N,)
        self.data[self.next] = x
        self.next = (self.next + 1) % self.maxlen

    def clear(self, dims: ep.Tensor) -> None:
        if self.tensor is None:
            self.tensor = dims  # pragma: no cover
        dims = dims.numpy()
        assert dims.shape == (self.N,)
        assert dims.dtype == np.bool
        self.data[:, dims] = np.nan

    def mean(self) -> ep.Tensor:
        assert self.tensor is not None
        result = np.nanmean(self.data, axis=0)
        return ep.from_numpy(self.tensor, result)

    def isfull(self) -> ep.Tensor:
        assert self.tensor is not None
        result = ~np.isnan(self.data).any(axis=0)
        return ep.from_numpy(self.tensor, result)
 

    
def get_projected_gradients(x_current, x_orig, label, surrogate_model):
    
    if surrogate_model is None:
        return None
    
    device = surrogate_model.device
    
    source_direction_ = x_orig - x_current
    source_direction = source_direction_.numpy() #maybe make it in another way
    source_norm = np.linalg.norm(source_direction)
    source_direction = source_direction / source_norm
    
    criterion = fb.criteria.Misclassification(torch.tensor([0], device=device))
    classes = criterion.labels
    loss_fn = get_loss_fn(surrogate_model, classes)
    
    x0, restore_type = ep.astensor_(x_current + 1e-2*source_direction_)
    
    _, gradients = value_and_grad(loss_fn, x0)
    gradients = gradients.numpy()
    #gradients = np.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Project the gradients.
    dot = np.vdot(gradients, source_direction)
    projected_gradient = gradients - dot * source_direction
    
    norm_ = np.linalg.norm(projected_gradient)
    if norm_ > 1e-5:
        projected_gradient /= norm_
        
    projected_gradient = (-1.) * projected_gradient
       
    return projected_gradient
    
    
    
def get_loss_fn(model: Model, labels: ep.Tensor) -> Callable[[ep.Tensor], ep.Tensor]:
    # can be overridden by users
    def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
        logits = model(inputs)
        return ep.crossentropy(logits, labels).sum()

    return loss_fn

def value_and_grad(loss_fn: Callable[[ep.Tensor], ep.Tensor], x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor]:
    return ep.value_and_grad(loss_fn, x)    


def draw_proposals(
    bounds: Bounds,
    originals: ep.Tensor,
    perturbed: ep.Tensor,
    unnormalized_source_directions: ep.Tensor,
    source_directions: ep.Tensor,
    source_norms: ep.Tensor,
    spherical_steps: ep.Tensor,
    source_steps: ep.Tensor,
    surrogate_model: Model
) -> Tuple[ep.Tensor, ep.Tensor]:
    # remember the actual shape
    shape = originals.shape
    assert perturbed.shape == shape
    assert unnormalized_source_directions.shape == shape
    assert source_directions.shape == shape

    # flatten everything to (batch, size)
    originals = flatten(originals)
    perturbed = flatten(perturbed)
    unnormalized_source_directions = flatten(unnormalized_source_directions)
    source_directions = flatten(source_directions)
    N, D = originals.shape

    assert source_norms.shape == (N,)
    assert spherical_steps.shape == (N,)
    assert source_steps.shape == (N,)

    # draw from an iid Gaussian (we can share this across the whole batch)
    eta = ep.normal(perturbed, (D, 1))

    # make orthogonal (source_directions are normalized)
    eta = eta.T - ep.matmul(source_directions, eta) * source_directions
    assert eta.shape == (N, D)
    
    pg_factor = 0.5
    
    if not surrogate_model is None:
        device = surrogate_model.device
        projected_gradient = get_projected_gradients(perturbed.reshape(shape), originals.reshape(shape), 0, surrogate_model)
        projected_gradient = projected_gradient.reshape((N, D))
        projected_gradient = torch.tensor(projected_gradient, device=device)
        
        projected_gradient, restore_type = ep.astensor_(projected_gradient)

        eta = (1. - pg_factor) * eta + pg_factor * projected_gradient
    
    # rescale
    norms = ep.norms.l2(eta, axis=-1)
    assert norms.shape == (N,)
    eta = eta * atleast_kd(spherical_steps * source_norms / norms, eta.ndim)

    # project on the sphere using Pythagoras
    distances = atleast_kd((spherical_steps.square() + 1).sqrt(), eta.ndim)
    directions = eta - unnormalized_source_directions
    spherical_candidates = originals + directions / distances

    # clip
    min_, max_ = bounds
    spherical_candidates = spherical_candidates.clip(min_, max_)

    # step towards the original inputs
    new_source_directions = originals - spherical_candidates
    assert new_source_directions.ndim == 2
    new_source_directions_norms = ep.norms.l2(flatten(new_source_directions), axis=-1)

    # length if spherical_candidates would be exactly on the sphere
    lengths = source_steps * source_norms

    # length including correction for numerical deviation from sphere
    lengths = lengths + new_source_directions_norms - source_norms

    # make sure the step size is positive
    lengths = ep.maximum(lengths, 0)

    # normalize the length
    lengths = lengths / new_source_directions_norms
    lengths = atleast_kd(lengths, new_source_directions.ndim)
    
    candidates = spherical_candidates + lengths * new_source_directions

    # clip
    candidates = candidates.clip(min_, max_)

    # restore shape
    candidates = candidates.reshape(shape)
    spherical_candidates = spherical_candidates.reshape(shape)
                                         
    return candidates, spherical_candidates
