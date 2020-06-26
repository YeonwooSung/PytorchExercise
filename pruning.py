import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def print_out_params(module):
    print('\n\nmodule.named_parameters():')
    print(list(module.named_parameters()))
    print('\n\nmodule.named_buffers():')
    print(list(module.named_buffers()))

if __name__ == "__main__":
    # generate a LeNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device=device)

    # Inspect a moule
    module = model.conv1
    print_out_params(module)

    # prune at random 30% of the connections in the parameter named weight in the conv1 layer
    print('\n\nprune at random 30% of the connections in the parameter named weight in the conv1 layer')
    prune.random_unstructured(module, name="weight", amount=0.3)

    print_out_params(module)

    print('\n\nmodule.weight:', module.weight)


    # we can now prune the bias too, to see how the parameters, buffers, hooks, and attributes of the module change
    print('\n\nprune the bias too')
    prune.l1_unstructured(module, name="bias", amount=3)

    print_out_params(module)
    print('\n\nmodule.bias:', module.bias)
