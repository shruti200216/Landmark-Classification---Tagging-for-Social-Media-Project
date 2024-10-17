import torch
import torch.nn as nn
import torch.optim


def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """

    # Instantiate an instance of CrossEntropyLoss
    loss = nn.CrossEntropyLoss()

    return loss


def get_optimizer(model, optimizer, learning_rate, momentum=None, weight_decay=None):
    """
    Get the optimizer for the model based on the specified parameters.

    Args:
        model: The model for which to get the optimizer.
        optimizer (str): The name of the optimizer. Supported values: 'sgd', 'adam', 'rmsprop'.
        learning_rate (float): The learning rate for the optimizer.
        momentum (float, optional): The momentum factor for SGD optimizer. Default is None.
        weight_decay (float, optional): The weight decay (L2 penalty) for the optimizer. Default is None.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """
    if optimizer == 'sgd':
        if momentum is None:
            raise ValueError("Momentum must be provided for SGD optimizer.")
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)


def test_get_loss():

    loss = get_loss()

    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"


def test_get_optimizer_type(fake_model):
    
    opt = get_optimizer(fake_model, optimizer="sgd", learning_rate=0.001, momentum=0.9, weight_decay=0.0)
    
    assert isinstance(opt, torch.optim.Optimizer)


def test_get_optimizer_is_linked_with_model(fake_model):
    
    opt = get_optimizer(fake_model, optimizer="sgd", learning_rate=0.001, momentum=0.9, weight_decay=0.0)
    
    assert opt.param_groups[0]['params'][0] is fake_model.parameters().__next__()


def test_get_optimizer_returns_adam(fake_model):
    
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.001, weight_decay=0.0)
    
    assert isinstance(opt, torch.optim.Adam)


def test_get_optimizer_sets_learning_rate(fake_model):
    
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123, weight_decay=0.0)
    
    assert opt.param_groups[0]['lr'] == 0.123


def test_get_optimizer_sets_momentum(fake_model):
    
    opt = get_optimizer(fake_model, optimizer="sgd", learning_rate=0.001, momentum=0.123, weight_decay=0.0)
    
    assert opt.param_groups[0]['momentum'] == 0.123


def test_get_optimizer_sets_weight_decay(fake_model):
    
    opt = get_optimizer(fake_model, optimizer="sgd", learning_rate=0.001, momentum=0.9, weight_decay=0.123)
    
    assert opt.param_groups[0]['weight_decay'] == 0.123