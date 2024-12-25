# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import jittor as jt
import jittor.nn as nn
import jittor.misc as misc
import numpy as np

from jittor.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, FocalLoss
from scipy import ndimage

import models.configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def _pair(value):
    """
    Converts the input to a tuple of two values.
    If the input is already a tuple, return it directly.
    Otherwise, create a tuple (value, value).
    """
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value, value)

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = jt.transpose(weights, 3, 2, 0, 1)
    return jt.float32(weights)

def swish(x):
    return x * x.sigmoid()

ACT2FN = {"gelu": nn.GELU(), "relu": nn.ReLU(), "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, loss_fct=None):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.loss_fct = loss_fct
        

    def execute(self, x, target):
        logprobs = nn.log_softmax(x, dim=-1)

        nll_loss = self.loss_fct(x, target)
        smooth_loss = -logprobs.mean(dim=-1).mean()
        return self.confidence * nll_loss + self.smoothing * smooth_loss

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = jt.size(x)[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = jt.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def execute(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = jt.matmul(query_layer, jt.transpose(key_layer, -1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = jt.matmul(attention_probs, value_layer)
        context_layer = misc.contiguous(context_layer.permute(0, 2, 1, 3))
        new_context_layer_shape = jt.size(context_layer)[:-2] + (self.all_head_size,)
        context_layer = jt.reshape(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        
        if self.config.psm_policy == "ffvt":
            attention_probs_col = self.softmax(jt.transpose(attention_scores, -1, -2)).transpose(-1, -2)
            last_map = attention_probs_col * attention_probs
            last_map = last_map[:,:,0,1:]
            max_inx, _ = jt.argmax(last_map, dim=2)
            return attention_output, attention_scores, max_inx
        return attention_output, attention_scores
class MlpConv(nn.Module):
    def __init__(self, config):
        super(MlpConv, self).__init__()
        self.hidden_size = config.hidden_size
        self.mlp_dim = config.transformer["mlp_dim"]
        self.h = 28
        self.w = 28

        self.fc1 = nn.Conv2d(self.hidden_size, self.mlp_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(self.mlp_dim, self.mlp_dim, kernel_size=3, padding=1, groups=self.mlp_dim)
        self.fc2 = nn.Conv2d(self.mlp_dim, self.hidden_size, kernel_size=1)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        jt.init.xavier_uniform_(self.fc1.weight)
        jt.init.xavier_uniform_(self.fc2.weight)
        jt.init.xavier_uniform_(self.depthwise_conv.weight)
        jt.init.gauss_(self.fc1.bias, std=1e-6)
        jt.init.gauss_(self.fc2.bias, std=1e-6)

    def execute(self, x):
        batch_size, seq_len, dim = x.shape
        h, w = self.h, self.w
        # assert (seq_len - 1) == h * w
        # 拆分类标记和图像标记
        cls_token, img_tokens = x[:, 0], x[:, 1:]
        img_tokens = img_tokens.view(batch_size, h, w, dim)

        img_tokens = img_tokens.permute(0, 3, 1, 2)  # (batch_size, dim, h, w)
        # 拆分类标记和图像标记
        cls_token, img_tokens = x[:, 0], x[:, 1:]
        img_tokens = img_tokens.permute(0, 2, 1).view(batch_size, dim, h, w)  # (batch_size, dim, h, w)

        img_tokens = self.fc1(img_tokens)
        img_tokens = self.depthwise_conv(img_tokens)
        img_tokens = self.act_fn(img_tokens)
        img_tokens = self.dropout(img_tokens)
        img_tokens = self.fc2(img_tokens)
        img_tokens = self.dropout(img_tokens)

        img_tokens = img_tokens.permute(0, 2, 3, 1).view(batch_size, seq_len - 1, dim)  # (batch_size, seq_len-1, dim)

        # 合并类标记和处理后的图像标记
        x = jt.concat([cls_token.unsqueeze(1), img_tokens], dim=1)  # (batch_size, seq_len, dim)
        return x
    
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        jt.init.xavier_uniform_(self.fc1.weight)
        jt.init.xavier_uniform_(self.fc2.weight)
        jt.init.gauss_(self.fc1.bias, std=1e-6)
        jt.init.gauss_(self.fc2.bias, std=1e-6)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = jt.zeros((1, n_patches+1, config.hidden_size))
        self.cls_token = jt.zeros((1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def execute(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x =jt.flatten(x, 2)
        x = jt.transpose(x, -1, -2)
        x = jt.concat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config, conv="use"):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.conv = conv
        if conv == "use":
            self.ffn = MlpConv(config)
        else:
            self.ffn = Mlp(config)
        self.attn = Attention(config)
        self.config = config

    def execute(self, x):
        h = x
        x = self.attention_norm(x)
        if self.config.psm_policy == "ffvt":
            x, scores, max_inx = self.attn(x)
        else:
            x, scores = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        if self.config.psm_policy == "ffvt":
            return x, scores, max_inx
        else:
            return x, scores

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with jt.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).transpose()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).transpose()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).transpose()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).transpose()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight = jt.copy(query_weight)
            self.attn.key.weight = jt.copy(key_weight)
            self.attn.value.weight = jt.copy(value_weight)
            self.attn.out.weight = jt.copy(out_weight)
            self.attn.query.bias = jt.copy(query_bias)
            self.attn.key.bias = jt.copy(key_bias)
            self.attn.value.bias = jt.copy(value_bias)
            self.attn.out.bias = jt.copy(out_bias)

            if self.conv != "use":
                mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).transpose()
                mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).transpose()
                mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).transpose()
                mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).transpose()

                self.ffn.fc1.weight = jt.copy(mlp_weight_0)
                self.ffn.fc2.weight = jt.copy(mlp_weight_1)
                self.ffn.fc1.bias = jt.copy(mlp_bias_0)
                self.ffn.fc2.bias = jt.copy(mlp_bias_1)

            self.attention_norm.weight = jt.copy(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias = jt.copy(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight = jt.copy(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias = jt.copy(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Part_Attention(nn.Module):
    def __init__(self, config):
        super(Part_Attention, self).__init__()
        self.config = config
        self.softmax = Softmax(dim=-1)
    def execute(self, x):
        attention_probs_row = [self.softmax(x_) for x_ in x]
        attention_probs_col = [self.softmax(x_.transpose(-1,-2)).transpose(-1,-2) for x_ in x]
        length = len(x)
        if self.config.psm_policy == "matmul":
            last_map = jt.init.eye(x[0].shape[-2:]).broadcast(x[0].shape)
            for i in range(0, length):
                if i in self.config.psm_layer:
                    last_map = jt.matmul(attention_probs_row[i], last_map)
        elif self.config.psm_policy == "dot":
            last_map = jt.ones_like(x[0])
            for i in range(0, length):
                if i in self.config.psm_layer:
                    last_map = attention_probs_row[i] * last_map
        elif self.config.psm_policy == "ffvt":
            last_map = jt.ones_like(x[0])
            for i in range(0, length):
                if i in self.config.psm_layer:
                    last_map = attention_probs_col[i] * attention_probs_row[i] * last_map
                
        last_map = last_map[:,:,0,1:]
        
        max_inx, _ = jt.argmax(last_map, dim=2)  # 最大值的索引
        map_weight = jt.ones((last_map.shape[0], last_map.shape[2]))
        for i in range(0, last_map.shape[1]):
            map_weight = map_weight * last_map[:, i, :]
        map_weight = (map_weight - jt.mean(map_weight, dim=-1, keepdims=True)) / jt.norm(map_weight, dim=-1, keepdims=True)
        return map_weight, max_inx

class Encoder(nn.Module):
    def __init__(self, config, psm=True):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.transformer["num_layers"] - 1):
            layer = Block(config, conv="no")
            self.layer.append(copy.deepcopy(layer))
        self.part_select = Part_Attention(config)
        self.part_layer = Block(config, conv="no")
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.psm = psm
        self.config = config

    def execute(self, hidden_states):
        attn_scores = []
        if self.psm:
            if self.config.psm_policy == "ffvt":
                final_parts = None
                for layer in self.layer:
                    parts = []
                    hidden_states, scores, part_inx = layer(hidden_states) 
                    B, num = part_inx.shape
                    for i in range(B):
                        parts.append(hidden_states[i, part_inx[i,:]])
                    parts = jt.stack(parts)
                    if final_parts is None:
                        final_parts = parts
                    else:
                        final_parts = jt.concat((final_parts, parts), dim=1)
                concat = jt.concat((hidden_states[:,0].unsqueeze(1), final_parts), dim=1)
                part_states, part_weights, max_inx = self.part_layer(concat)
                part_encoded = self.part_norm(part_states)   

                return part_encoded 
            else:
                for layer in self.layer:
                    hidden_states, scores = layer(hidden_states)
                    attn_scores.append(scores)        
                map_weight, part_inx = self.part_select(attn_scores)

                part_inx = part_inx + 1
                parts = []
                if self.config.select_policy == "discrete":
                    B, num = part_inx.shape
                    for i in range(B):
                        parts.append(hidden_states[i, part_inx[i,:]])
                    parts = jt.stack(parts)
                    concat = jt.concat((hidden_states[:,0].unsqueeze(1), parts), dim=1)
                    part_states, part_weights = self.part_layer(concat)
                elif self.config.select_policy == "continuous":
                    hidden_states[:, 1:, :] = hidden_states[:, 1:, :] * map_weight.unsqueeze(2)
                    part_states, part_weights = self.part_layer(hidden_states) 
                part_encoded = self.part_norm(part_states)   

                return part_encoded
        else:
            for layer in self.layer:
                hidden_states, scores = layer(hidden_states)  

            return hidden_states
        
class Transformer(nn.Module):
    def __init__(self, config, img_size, psm=True):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, psm)

    def execute(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        part_encoded = self.encoder(embedding_output)
        return part_encoded


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False, loss_function="CrossEntropyLoss", mix=False, clip_alpha=0.4, psm=True, con_loss=True):
        super(VisionTransformer, self).__init__()
        self.mix = mix
        self.loss_function = loss_function
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, psm)
        self.part_head = Linear(config.hidden_size, num_classes)
        self.clip_alpha=clip_alpha
        self.psm=psm
        self.con_loss=con_loss

    def execute(self, x, labels_a=None, labels_b=None, lam=1):
        part_tokens = self.transformer(x)
        part_logits = self.part_head(part_tokens[:, 0])

        if self.mix == 'none':
            if labels_a is not None:
                if self.loss_function == "FocalLoss":
                    loss_fct = FocalLoss()
                elif self.loss_function == "CrossEntropyLoss":
                    loss_fct = CrossEntropyLoss()
                loss_fct = LabelSmoothing(self.smoothing_value, loss_fct)
                loss = loss_fct(part_logits.view(-1, self.num_classes), labels_a.view(-1))
                if self.con_loss:
                    contrast_loss = con_loss(part_tokens[:, 0], labels_a.view(-1), self.clip_alpha)
                    loss += contrast_loss
                
                return loss, part_logits
            else:
                return part_logits
        else:
            if labels_a is not None and labels_b is not None: 
                if self.loss_function == "FocalLoss":
                    loss_fct = FocalLoss()
                elif self.loss_function == "CrossEntropyLoss":
                    loss_fct = CrossEntropyLoss()
                loss_fct = LabelSmoothing(self.smoothing_value, loss_fct)
                part_loss_a = loss_fct(part_logits.view(-1, self.num_classes), labels_a.view(-1))
                part_loss_b = loss_fct(part_logits.view(-1, self.num_classes), labels_b.view(-1))
                loss = lam * part_loss_a + (1-lam) * part_loss_b
                if self.con_loss:
                    contrast_loss = con_loss_mix(part_tokens[:, 0], labels_a.view(-1), labels_b.view(-1), lam, part_logits, self.clip_alpha)
                    loss += contrast_loss
                    
                return loss, part_logits
            else:
                return part_logits
            

    def load_from(self, weights):
        with jt.no_grad():
            self.transformer.embeddings.patch_embeddings.weight = jt.copy(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias = jt.copy(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token = jt.copy(np2th(weights["cls"]))
            self.transformer.encoder.part_norm.weight = jt.copy(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.part_norm.bias = jt.copy(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings = jt.copy(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = jt.reshape(posemb_grid, (gs_old, gs_old, -1))

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = jt.reshape(posemb_grid, (1, gs_new * gs_new, -1))
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings = jt.copy(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('part') == False:
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                print(1)
                self.transformer.embeddings.hybrid_model.root.conv.weight = jt.copy(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight = jt.copy(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias = jt.copy(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) 
            else:
                print(2)

def con_loss(features, labels, clip_alpha):
    B, _ = features.shape
    features = misc.normalize(features)
    cos_matrix = jt.matmul(features, features.transpose())
    pos_label_matrix = jt.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - clip_alpha
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

def con_loss_mix(features, labels_a, labels_b, lam, part_logits, clip_alpha):
    B, _ = features.shape
    features = misc.normalize(features)
    cos_matrix = jt.matmul(features, features.transpose())
    pos_label_matrix = jt.stack([labels_a == labels_a[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - clip_alpha
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)

    pos_label_matrix = jt.stack([labels_b == labels_b[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - clip_alpha
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss2 = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss2 /= (B * B)
    return lam * loss + (1 - lam) * loss2


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
