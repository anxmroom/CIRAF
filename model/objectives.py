import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_sdmid(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):

    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)


    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i + loss_t) / 2

    return loss

class Attribute(nn.Module):

    def __init__(self):
        super().__init__()

    def compote_attribute_loss(self, attributes, text_feats, Vgs):
        loss_attribute = 0
        count = 0
        for attribute, text_feat, Vg in zip(attributes, text_feats, Vgs):
            max_attribute_value = torch.max(attribute).to(torch.int32)

            averaged_attribute = []

            for i in range(1, max_attribute_value + 1):
                mask = attribute == i
                averaged_attribute.append(text_feat[mask].mean(0))

            if len(averaged_attribute) > 0:
                total_attribute = torch.stack(averaged_attribute)
                cosine_similarity = nn.functional.cosine_similarity(Vg, total_attribute, dim=1).mean(0)
                loss_attribute += (1 - cosine_similarity)
                count += 1
        loss_attribute = loss_attribute / count
        return loss_attribute
    def compote_neg_attribute_loss(self, attributes, text_feats, Vgs, text_neg_idxs):
        loss_attribute = 0
        count = 0
        for attribute, Vg, text_neg_idx in zip(attributes, Vgs, text_neg_idxs):
            max_attribute_value = torch.max(attribute).to(torch.int32)
            text_feat = text_feats[text_neg_idx]
            averaged_attribute = []

            for i in range(1, max_attribute_value + 1):
                mask = attribute == i
                averaged_attribute.append(text_feat[mask].mean(0))

            if len(averaged_attribute) > 0:
                total_attribute = torch.stack(averaged_attribute)
                cosine_similarity = nn.functional.cosine_similarity(Vg, total_attribute, dim=1).mean(0)
                loss_attribute += (1 - cosine_similarity)
                count += 1
        loss_attribute = loss_attribute / count
        return loss_attribute


