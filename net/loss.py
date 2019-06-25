import torch
import torch.nn.functional as F
import torch.nn as nn
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
    pred_ = pred[label_inds] 
    #print(gt.size(0))
    #print(weight)
    weight = weight.repeat(gt.size(0),1)
    pos_loss = torch.log(pred_) * torch.pow(1 - pred_, 2) * weight[label_inds]    

    num  = label_inds.float().sum()
    pos_loss = pos_loss.sum()
    loss = pos_loss/ num
    #print(loss)
    return -loss


# class PersonAttr_Loss(nn.Module):
#     def __init__(self, pos_weight ):
#         super(PersonAttr_Loss, self).__init__()

#         self.backpack_weight = pos_weight[0]
#         self.bag_weight = pos_weight[1]
#         self.handbag_weight  = pos_weight[2]
#         self.down_weight  = pos_weight[3]
#         self.up_weight  = pos_weight[4]
#         self.hair_weight  = pos_weight[5]
#         self.hat_weight     = pos_weight[6]
#         self.gender_weight     = pos_weight[7]
#         self.upcolor_weight =  pos_weight[8:17]
#         self.downcolor_weight =  pos_weight[17:24]
#     def forward(self, outs, targets):
#         #print(outs)
#         backpack = _sigmoid(outs[0])
#         bag = _sigmoid(outs[1])
#         handbag  = _sigmoid(outs[2])
#         down  = _sigmoid(outs[3])
#         up  = _sigmoid(outs[4])
#         hair  = _sigmoid(outs[5])
#         hat     = _sigmoid(outs[6])
#         gender     = _sigmoid(outs[7])
#         upcolor = outs[8]
#         downcolor = outs[9]

#         gt_backpack  = targets[:,0]
#         gt_bag  = targets[:,1]
#         gt_handbag     = targets[:,2]
#         gt_down   = targets[:,3]
#         gt_up   = targets[:,4]
#         gt_hair   = targets[:,5]
#         gt_hat   = targets[:,6]
#         gt_gender = targets[:,7]

#         gt_upcolor = targets[:,8:17]
#         gt_downcolor     = targets[:,17:24]

#         # class loss
#         backpack_loss =  _clc_loss_mask(backpack,gt_backpack,self.backpack_weight)
#         bag_loss =  _clc_loss_mask(bag,gt_bag,self.bag_weight)
#         handbag_loss =  _clc_loss_mask(handbag,gt_handbag,self.handbag_weight)
#         down_loss =  _clc_loss_mask(down,gt_down,self.down_weight)
#         up_loss =  _clc_loss_mask(up,gt_up,self.up_weight)
#         hair_loss =  _clc_loss_mask(hair,gt_hair,self.hair_weight)
#         hat_loss =  _clc_loss_mask(hat,gt_hat,self.hat_weight)
#         gender_loss =  _clc_loss_mask(gender,gt_gender,self.gender_weight)

#         #down color
#         downcolor_loss = _softmax_loss(downcolor,gt_downcolor,self.downcolor_weight)

#         #upcolor
#         upcolor_loss = _softmax_loss(upcolor,gt_upcolor,self.upcolor_weight)

#         loss = backpack_loss + bag_loss +  handbag_loss + down_loss + up_loss + hair_loss + hat_loss + gender_loss + downcolor_loss + upcolor_loss
#         #print(loss)
#         return loss.unsqueeze(0)


class PersonAttr_Loss(nn.Module):
    def __init__(self, pos_weight ):
        super(PersonAttr_Loss, self).__init__()
        self.bag_weight = pos_weight[0]
        self.handbag_weight  = pos_weight[1]
        self.down_weight  = pos_weight[2]
        self.up_weight  = pos_weight[3]
        self.hair_weight  = pos_weight[4]
        self.hat_weight     = pos_weight[5]
        self.gender_weight     = pos_weight[6]
        self.upcolor_weight =  pos_weight[7:16]
        self.downcolor_weight =  pos_weight[16:23]
    def forward(self, outs, targets):
        #print(outs)
        bag = _sigmoid(outs[0])
        handbag  = _sigmoid(outs[1])
        down  = _sigmoid(outs[2])
        up  = _sigmoid(outs[3])
        hair  = _sigmoid(outs[4])
        hat     = _sigmoid(outs[5])
        gender     = _sigmoid(outs[6])
        upcolor = outs[7]
        downcolor = outs[8]

        gt_bag  = targets[:,0]
        gt_handbag     = targets[:,1]
        gt_down   = targets[:,2]
        gt_up   = targets[:,3]
        gt_hair   = targets[:,4]
        gt_hat   = targets[:,5]
        gt_gender = targets[:,6]

        gt_upcolor = targets[:,7:16]
        gt_downcolor     = targets[:,16:23]

        # class loss

        bag_loss =  _clc_loss_mask(bag,gt_bag,self.bag_weight)
        handbag_loss =  _clc_loss_mask(handbag,gt_handbag,self.handbag_weight)
        down_loss =  _clc_loss_mask(down,gt_down,self.down_weight)
        up_loss =  _clc_loss_mask(up,gt_up,self.up_weight)
        hair_loss =  _clc_loss_mask(hair,gt_hair,self.hair_weight)
        hat_loss =  _clc_loss_mask(hat,gt_hat,self.hat_weight)
        gender_loss =  _clc_loss_mask(gender,gt_gender,self.gender_weight)

        #down color
        downcolor_loss = _softmax_loss(downcolor,gt_downcolor,self.downcolor_weight)

        #upcolor
        upcolor_loss = _softmax_loss(upcolor,gt_upcolor,self.upcolor_weight)

        loss = bag_loss +  handbag_loss + down_loss + up_loss + hair_loss + hat_loss + gender_loss + downcolor_loss + upcolor_loss
        #print(loss)
        return loss.unsqueeze(0)