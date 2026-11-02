"""The linear stack really is one weighted sum. The ReLU stack is not."""

import torch

from models import DeepLinearNet, LinearNet, NonlinearNet, count_parameters


def test_linear_and_relu_have_the_same_weights():
    assert count_parameters(LinearNet()) == count_parameters(NonlinearNet())


def test_stacked_linear_layers_match_one_affine_map():
    torch.manual_seed(1)
    model = LinearNet()
    inputs = torch.randn(40, 2)
    with torch.no_grad():
        output = model(inputs)
        hidden = inputs @ model.hidden.weight.T + model.hidden.bias
        manual = hidden @ model.output.weight.T + model.output.bias
    assert torch.allclose(output, manual)


def test_deep_linear_stack_is_still_one_affine_map():
    torch.manual_seed(2)
    model = DeepLinearNet()
    inputs = torch.randn(20, 2)
    with torch.no_grad():
        output = model(inputs)
        weight = model.layer3.weight @ model.layer2.weight @ model.layer1.weight
        bias = model.layer3.weight @ (
            model.layer2.weight @ model.layer1.bias + model.layer2.bias
        )
        bias = bias + model.layer3.bias
        manual = inputs @ weight.T + bias
    assert torch.allclose(output, manual, atol=1e-5)


def test_relu_changes_the_result_of_the_same_weights():
    model = NonlinearNet()
    with torch.no_grad():
        model.hidden.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            )
        )
        model.hidden.bias.zero_()
        model.output.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
        model.output.bias.zero_()
        inputs = torch.tensor([[1.0, 0.0]])
        bent = model(inputs)
        straight = model.output(model.hidden(inputs))
    assert bent.item() == 1
    assert straight.item() == 0
