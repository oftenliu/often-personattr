import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-6, max=1-1e-6)


def _clc_loss_mask(pred, gt, pos_weight):
    pos_inds = gt.eq(1)
    neg_inds = gt.eq(0)

    #unknow = gt.lt(0)

    loss = 0

    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * pos_weight
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * (1-pos_weight)

    num_pos  = pos_inds.float().sum()
    num_neg  = neg_inds.float().sum()


    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()


    if num_pos + num_neg == 0:
        loss = 0
    else:
        loss = (pos_loss + neg_loss) / (num_pos+num_neg)
    #print(loss)
    return -loss


def _softmax_loss(pred, gt,weight):
    
    label_inds = gt.eq(1)   
    num  = label_inds.float().sum()
    if num == 0:
        return 0
    pred_ = pred[label_inds] 
    #print(gt.size(0))
    #print(weight)
    weight = weight.repeat(gt.size(0),1)
    #print(weight)
    pos_loss = torch.log(pred_) * torch.pow(1 - pred_, 2) * weight[label_inds]    


    pos_loss = pos_loss.sum()
    loss = pos_loss/ num
    #print(loss)
    return -loss

# class_value = [1,5,1,1,1, 1, 1,1, 1,1, 1, 1,1,1, 1, 1, 1,1 ,1, 1, 1,1, 1, 1,1, 1,               
#                1, 1,1, 1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1,1, 1, 1,
#                1, 1, 1,1, 1, 1,1,
# ]

class PersonAttr_Loss(nn.Module):
    def __init__(self, pos_dist,class_value,class_name ):
        super(PersonAttr_Loss, self).__init__()
        self._pos_dist = pos_dist #正样本分布
        self.class_value = class_value
        self.class_name = class_name
    def forward(self, outs, targets):
        #print(len(outs))
        gt_base = 0
        loss = 0
        #print(len(outs),len(self.class_name),len(self.class_value),len(targets[0]))
        pred_success = 0
        label_number = 0
        for idx in range(0,len(self.class_value)):
            if self.class_value[idx] == 1:
                out = _sigmoid(outs[idx])
                gt  = targets[:,gt_base+idx]
                #print("class 1" + str(gt_base+idx))
                loss = loss + _clc_loss_mask(out,gt,1 - self._pos_dist[gt_base+idx])
                preds = torch.gt(out, torch.ones_like(out)/2 )
                preds = preds.squeeze(1)
                pred_success = pred_success + torch.sum(preds == gt.data.byte()).item()
                label_number += 1
            elif idx == 1: #age
                out = outs[idx]
                gt     = targets[:,gt_base+idx:gt_base+idx+self.class_value[idx]]
                
                #print("class " + str(class_value[idx]) + str(gt_base+idx) + "~" + str(gt_base+idx+5))

                weight = 1- self._pos_dist[gt_base+idx:gt_base+idx+self.class_value[idx]]
                loss = loss + _softmax_loss(out,gt,weight)

                gt_cls = torch.max(gt ,1)[1].data.byte()
                output_cls = torch.max(out,1)[1].data.byte()
                pred_success = pred_success + torch.sum(gt_cls == output_cls.data.byte()).item() 
                gt_base = gt_base + self.class_value[idx] - 1
                label_number += 1
            else:
                out = outs[idx]
                gt     = targets[:,gt_base+idx:gt_base+idx+self.class_value[idx]]
                weight = self._pos_dist[gt_base+idx:gt_base+idx+self.class_value[idx]]
                #print(weight)
                weights_label = torch.zeros(gt.shape)
                for i in range(gt.shape[0]):
                    for j in range(gt.shape[1]):
                        if gt.data.cpu()[i, j] == 1:
                            weights_label[i, j] = 1 - weight[j]
                        else:
                            weights_label[i, j] = weight[j]
                weights_label = weights_label.cuda()
                #print(weights_label)
                loss = loss + F.binary_cross_entropy_with_logits(out,gt,weights_label)
                # for subclass_id in range(0,self.class_value[idx]):
                #     out_subclass = _sigmoid(out[:,subclass_id])
                    
                #     gt_subclass  = gt[:,subclass_id]
                #     #print("class 1" + str(gt_base+idx))
                #     loss = loss + _clc_loss_mask(out_subclass,gt_subclass,wieght[subclass_id])
                preds = torch.gt(out, torch.ones_like(out)/2 )
                #     #print(preds)
                #     #preds = preds.squeeze(1)
                pred_success = pred_success + torch.sum(preds == gt.data.byte()).item() 
                gt_base = gt_base + self.class_value[idx] - 1      
                label_number += len(out[0])         
        #print(label_number)
        return loss.unsqueeze(0),pred_success/label_number