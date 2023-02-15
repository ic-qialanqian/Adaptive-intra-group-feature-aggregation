from torch import nn
import torch
import math

CE = torch.nn.BCEWithLogitsLoss()

def iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    #for i in range(0,b):
        #compute the IoU of the foreground
    Iand1 = torch.sum(target*pred)
    Ior1 = torch.sum(target) + torch.sum(pred)-Iand1
    IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        #IoU = IoU + (1-IoU1)

    return IoU1

class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        
        IoU = 0.0
        for i in range(0,b):
            #compute the IoU of the foreground
            Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
            Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
            IoU1 = Iand1/Ior1

            #IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        
        #return IoU/b
        return IoU

class Scale_loss(nn.Module):
    def __init__(self):
        super(Scale_loss, self).__init__()
        self.iou = IoU_loss()

    def forward(self, scaled_preds, gt):
        loss = 0
        for pred_lvl in scaled_preds[1:]:
            f_end = int(gt.shape[0] / 2)
            s_end = int(gt.shape[0])
            
            
            alpha1= 1-iou(pred_lvl[0:f_end,:,:,:], gt[0:f_end,:,:,:])
            alpha2= 1-iou(pred_lvl[f_end:s_end,:,:,:], gt[f_end:s_end,:,:,:])
           
            
            num_gt1 = len(torch.nonzero(gt[0:f_end,:,:,:]))
            num_gt2 = len(torch.nonzero(gt[f_end:s_end,:,:,:]))
            
            
            group1 = num_gt1/(50176*f_end)
            group2 = num_gt2/(50176*f_end)
            
            s1 = 1-num_gt1/(50176*f_end)
            s2 = 1-num_gt2/(50176*f_end)
            
           
            
           
            beta1 = s1 ** 0.1
            beta2 = s2 ** 0.1
            
            
            
            loss += self.iou(pred_lvl[0:f_end,:,:,:], gt[0:f_end,:,:,:]) * beta1 + self.iou(pred_lvl[f_end:s_end,:,:,:], gt[f_end:s_end,:,:,:]) * beta2
            
        return loss
        

