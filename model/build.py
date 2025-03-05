from collections import OrderedDict

from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, QuickGELU
import torch
import torch.nn as nn
from datasets.bases import AttentionMask
import torch.nn.functional as F
from model.objectives import Attribute_loss, Attribute
import numpy as np

class CIRAF(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)

        self.img_k_ratio = 0.2
        self.text_k_ratio = 0.2

        self.embed_dim = base_cfg['embed_dim']
        self.transformer_width = base_cfg['transformer_width']
        self.num_x=(base_cfg['image_resolution'][1] - base_cfg['vision_patch_size'] )// base_cfg['stride_size'] + 1
        self.num_y = (base_cfg['image_resolution'][0] - base_cfg['vision_patch_size']) // base_cfg['stride_size'] + 1
        self.key_image_width = int((self.num_x * self.num_y+1)*self.img_k_ratio)
        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.attn_mask = AttentionMask()

        if 'ATT' in args.loss_names:
            self.attribute_loss = Attribute()

        if 'amlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

                # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                    OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                                    ('gelu', QuickGELU()),
                                    ('ln', LayerNorm(self.embed_dim)),
                                    ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

            self.conv = nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                              kernel_size=self.key_image_width)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q).float(),
            self.ln_pre_i(k).float(),
            self.ln_pre_i(v).float(),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)
        x = self.cross_modal_transformer(x)[0]
        x = x.permute(1, 0, 2)

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[0][:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)[0]
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def select_image_feats(self, image_feats, image_score):
        # selcet image feats
        image_score = image_score[-1][:, 0, :]  # [48,193] 48是一个batch中有48张图，每个batch中即每张图中第一个patch即全局对所有patch的得分
        image_feats_selected_all = []
        img_K = int(image_feats.size(1) * self.img_k_ratio)
        for b in range(image_feats.size(0)):
            _, idx = image_score[b].topk(image_score.size(1), largest=True, sorted=True)
            image_feats_selected_all.append(image_feats[b][idx[:img_K], :])  # 逗号后面规定取对少最后一维
        image_feats_selected_all = torch.stack(image_feats_selected_all, dim=0)
        return image_feats_selected_all

    def forward(self, batch):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids']
        mlm_ids = batch['mlm_ids']
        image_feats, text_feats = self.base_model(images, mlm_ids)
        #2.att_text_feats = self.base_model.encode_text(caption_ids)[0]

        image_feats, image_score = image_feats
        text_feats, text_score = text_feats

        Tg = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)]
        Vg = image_feats[:, 0, :].half()

        logit_scale = self.logit_scale

        image_feats_selected_all = self.select_image_feats(image_feats, image_score)

        # select attention masked text id
        text_score = text_score[-1][torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)]

        textmlm_id = []
        temp = torch.zeros(text_score.size(0), 1)
        score_mask_text = (torch.cat((temp, torch.ones(text_score.size(0), text_score.size(1) - 1)), dim=1)).to(
            text_score.device)
        score_mask_text[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)] = 0
        text_score = text_score * score_mask_text
        text_K = torch.round((caption_ids.argmax(dim=-1)+1) * self.text_k_ratio).int()
        for b in range(text_feats.size(0)):
            _, idx = text_score[b].topk(text_score.size(1), largest=True, sorted=True)
            textmlm_id.append(idx[:text_K[b]])

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(Vg, Tg, logit_scale)})

        if 'ATT' in self.current_task:
            image_norm = Vg / Vg.norm(dim=1, keepdim=True)
            text_norm = Tg / Tg.norm(dim=1, keepdim=True)
            sim_i2t = image_norm @ text_norm.t()
            weights_i2t = F.softmax(sim_i2t, dim=1)
            mask = torch.eq(batch['pids'].unsqueeze(1), batch['pids'].unsqueeze(1).T)
            weights_i2t.masked_fill_(mask, 0)
            text_neg_idx = torch.multinomial(weights_i2t, 1).flatten()

            positive_attribute = batch['attribute_mask']
            negative_attribute = batch['attribute_mask'][text_neg_idx]
            attribute_loss1 = self.attribute_loss.compote_attribute_loss(positive_attribute, text_feats, Vg)
            attribute_loss2 = self.attribute_loss.compote_neg_attribute_loss(negative_attribute, text_feats, Vg, text_neg_idx)
            ret.update({'attribute_loss': ((attribute_loss1+attribute_loss2)*self.args.att_loss_weight)})

        if 'msdm' in self.current_task:
            ret.update({'sdmid_loss': objectives.compute_sdmid(Vg, Tg, batch['pids'], logit_scale)})

        if 'amlm' in self.current_task:
            mlm_ids, mlm_labels = self.attn_mask._build_attention_masked_tokens_and_labels(caption_ids, textmlm_id)
            mlm_ids = torch.stack(mlm_ids, dim=0)
            mlm_labels = torch.tensor(mlm_labels, device=mlm_ids.device)
            mlm_feats = self.base_model.encode_text(mlm_ids)[0]
            x = self.cross_former(mlm_feats, image_feats_selected_all, image_feats_selected_all)
            scores = self.mlm_head(x).reshape(-1, self.args.vocab_size)
            mlm_labels = mlm_labels.reshape(-1)
            ret.update({'amlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'amlm_acc': acc})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)[0]

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})


        ret.update({'temperature': 1 / logit_scale})
        return ret


def build_model(args, num_classes=11003):
    model = CIRAF(args, num_classes)
    return model
