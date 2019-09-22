"""

"""


# Built-in

# Libs

# Pytorch
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


# Own modules


def to_one_hot_var(tensor, nClasses, requires_grad=False):
    try:
        n, h, w = tensor.size()
    except ValueError:
        n, _, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)
    return Variable(one_hot, requires_grad=requires_grad)


class LossClass(nn.Module):
    """
    The base class of loss metrics, all loss metrics should inherit from this class
    This class contains a function that defines how loss is computed (def forward) and a loss tracker that keeps
    updating the loss within an epoch
    """
    def __init__(self):
        super(LossClass, self).__init__()
        self.loss = 0
        self.cnt = 0

    def forward(self, pred, lbl):
        raise NotImplementedError

    def update(self, loss, size):
        """
        Update the current loss tracker
        :param loss: the computed loss
        :param size: #elements in the batch
        :return:
        """
        self.loss += loss.item() * size
        self.cnt += 1

    def reset(self):
        """
        Reset the loss tracker
        :return:
        """
        self.loss = 0
        self.cnt = 0

    def get_loss(self):
        """
        Get mean loss within this epoch
        :return:
        """
        return self.loss / self.cnt


class CrossEntropyLoss2d(LossClass):
    """
    Cross entropy loss function used in training
    """

    def __init__(self, weight=None, size_average=True, ignore_index=255, reduce=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.name = 'Angle'
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index, reduce)

    def forward(self, inputs, targets):
        log_p = F.log_softmax(inputs, dim=1)
        loss = self.nll_loss(log_p, torch.squeeze(targets, dim=1))
        return loss


class mIoULoss(LossClass):
    def __init__(self, weight=None, n_classes=2):
        super(mIoULoss, self).__init__()
        self.name = 'Road'
        self.classes = n_classes
        self.weights = Variable(weight * weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weights * inter) / (self.weights * union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)
