import torch
import torch.nn as nn

loss_names = ['l1', 'l2', 'l1_s12', 'l2_s12']
rgb_loss_names = ['l1', 'l2']


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        if mask is not None:
            diff = diff[mask.expand(diff.shape)]
        self.loss = (diff**2).mean()
        return self.loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, pred, target, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target - pred
        if mask is not None:
            diff = diff[mask.expand(diff.shape)]
        self.loss = diff.abs().mean()
        return self.loss

class MSES12Loss(nn.Module):
    def __init__(self):
        super(MSES12Loss, self).__init__()

    def forward(self, pred, target, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target[:,1:3,:,:] - pred[:,1:3,:,:]
        if mask is not None:
            diff = diff[mask.expand(diff.shape)]
        self.loss = (diff**2).mean()
        return self.loss

class L1S12Loss(nn.Module):
    def __init__(self):
        super(L1S12Loss, self).__init__()

    def forward(self, pred, target, mask=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        diff = target[:,1:3,:,:] - pred[:,1:3,:,:]
        if mask is not None:
            diff = diff[mask.expand(diff.shape)]
        self.loss = diff.abs().mean()
        return self.loss