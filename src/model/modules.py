import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bart import *
from src.model.modeling_bart import (
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    invert_mask,
    EncoderLayer,
    LayerNorm,
)
from src.model.modeling_bart import (PretrainedBartModel, BartDecoder,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from src.model.config import MultiModalBartConfig
import torch.nn.functional as F
import pdb
import copy

class ImageEmbedding(nn.Module):
    def __init__(self, image_dim, final_dim):
        super(ImageEmbedding, self).__init__()
        self.linear = nn.Linear(image_dim, final_dim)

    def forward(self, image_features):
        img_len = list(map(len, image_features))
        non_empty_features = list(filter(lambda x: len(x) > 0, image_features))

        embedded = None
        if len(non_empty_features) > 0:
            img_tensor = torch.cat(non_empty_features, dim=0)
            embedded = self.linear(img_tensor)

        output = []
        index = 0
        for l in img_len:
            if l > 0:
                output.append(embedded[index:index + l])
            else:
                output.append(torch.empty(0))
            index += l
        return output


class MultiModalBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:EncoderLayer.

    Args:
        config: MultiModalBartConfig
    """
    def __init__(self, config: MultiModalBartConfig, encoder, img_feat_id,
                 cls_token_id):
        super().__init__()

        self.img_feat_id = img_feat_id
        self.cls_token_id = cls_token_id
        embed_tokens = encoder.embed_tokens
        self.dropout = encoder.dropout
        self.layerdrop = encoder.layerdrop

        self.indentity = nn.Identity()

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = encoder.embed_scale
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = encoder.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_images = ImageEmbedding(2048, embed_dim)
        self.embed_positions = encoder.embed_positions

        self.layers = encoder.layers
        self.layernorm_embedding = encoder.layernorm_embedding
        # mbart has one extra layer_norm
        self.layer_norm = encoder.layer_norm

    def _embed_multi_modal(self, input_ids, image_features):
        """embed textual and visual inputs and combine them into one embedding"""
        mask = (input_ids == self.img_feat_id) | (
            input_ids == self.cls_token_id)
        embedded_images = self.embed_images(image_features)
        embedded = self.embed_tokens(input_ids)
        if not embedded_images[0].dtype == torch.float32:
            embedded = embedded.half()
        for index, value in enumerate(embedded_images):
            if len(value) > 0:
                embedded[index, mask[index]] = value
        return embedded

    def forward(self,
                input_ids,
                image_features,
                attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False):
        """

        :param input_ids: LongTensor, tokens in the source language of shape (batch, src_len)
        :param image_features: list[FloatTensor], image roi features with length of batch
        :param attention_mask: LongTensor, indicating which indices are padding tokens.
        :param output_attentions:
        :param output_hidden_states:
        :return: Tuple comprised of:
            - x (Tensor): the last encoder layer's output of
              shape (src_len, batch, embed_dim)
            - encoder_states (List[Tensor]): all intermediate
              hidden states of shape (src_len, batch, embed_dim).
              Only populated if output_hidden_states: is True.
            - all_attentions (List[Tensor]): Attention weights for each layer.
            During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self._embed_multi_modal(
            input_ids, image_features) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability <
                                  self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x,
                                        attention_mask,
                                        output_attentions=output_attentions)

            if output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [
            hidden_state.transpose(0, 1) for hidden_state in encoder_states
        ]
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(last_hidden_state=x,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


class MultiModalBartDecoder_span(nn.Module
                                 ):  #AOE task and all downstream tasks
    def __init__(self,
                 config: MultiModalBartConfig,
                 tokenizer,
                 decoder,
                 pad_token_id,
                 label_ids,
                 causal_mask,
                 gcn_on,
                 need_tag=True,
                 only_sc=False,
                 avg_feature=False,
                 use_encoder_mlp=True):
        super().__init__()
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.causal_mask = causal_mask
        self.gcn_on=gcn_on

        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        # label_ids = sorted(label_ids, reverse=False)
        self.label_start_id = min(label_ids)
        self.label_end_id = max(label_ids) + 1
        self.need_tag = need_tag
        self.only_sc = only_sc
        mapping = torch.LongTensor([0, 2] + label_ids)
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.dropout_layer = nn.Dropout(0.1)

        self.end_text_id = tokenizer.end_text_id
        self.avg_feature = avg_feature
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.Dropout(0.3),
                nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.senti_value_linear=nn.Linear(1,768)


    def forward(self, tokens, state,sentiment_value,only_sc=False):
        if self.gcn_on:
            encoder_pad_mask = state.encoder_mask
            mix_feature=state.mix_feature

            first = state.first
            # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
            cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
            tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

            # 把输入做一下映射
            mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
            mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
            tag_mapped_tokens = self.mapping[mapped_tokens]
            src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
            src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
            src_tokens = state.src_tokens
            if first is not None:
                src_tokens = src_tokens.gather(index=first, dim=1)
            word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
            tokens = torch.where(mapping_token_mask, tag_mapped_tokens,word_mapped_tokens)
            tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
            if self.training:
                tokens = tokens[:, :-1]
                decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
                dict = self.decoder(input_ids=tokens,
                                            encoder_hidden_states=mix_feature,
                                            encoder_padding_mask=encoder_pad_mask,
                                            decoder_padding_mask=decoder_pad_mask,
                                            decoder_causal_mask=self.
                                            causal_masks[:tokens.size(1), :tokens.size(1)],
                                            return_dict=True)

            else:
                past_key_values = state.past_key_values
                dict = self.decoder(input_ids=tokens,
                                            encoder_hidden_states=mix_feature,
                                            encoder_padding_mask=encoder_pad_mask,
                                            decoder_padding_mask=None,
                                            decoder_causal_mask=self.
                                            causal_masks[:tokens.size(1), :tokens.size(1)],
                                            return_dict=True)

            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
            hidden_state = self.dropout_layer(hidden_state)
            if not self.training:
                state.past_key_values = dict.past_key_values

            logits = hidden_state.new_full(
                (hidden_state.size(0), hidden_state.size(1),
                 self.src_start_index + src_tokens.size(-1)),
                fill_value=-1e24)
            # 首先计算的是

            if self.need_tag:   #if predict the sentiment or not
                # if hidden_state.shape[0]!=sentiment_value.shape[0]:
                #     # 预测阶段
                #     indices = torch.arange(sentiment_value.shape[0], dtype=torch.long,device=sentiment_value.device)
                #     indices = indices.repeat_interleave(4)
                #     sentiment_value=sentiment_value.index_select(dim=0,index=indices)
                # sentiment_value=torch.tanh(self.senti_value_linear(sentiment_value.sum(dim=-1).unsqueeze(-1))).unsqueeze(1).repeat(1,hidden_state.shape[1],1)
                tag_scores = F.linear(
                    hidden_state,
                    self.dropout_layer(
                        self.decoder.embed_tokens.
                        weight[self.label_start_id:self.label_start_id +
                                                   3]))  # bsz x max_len x num_class   相当于在计算向量相似度
                logits[:, :, 3:self.src_start_index] = tag_scores


            if not only_sc:
                eos_scores = F.linear(
                    hidden_state,
                    self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))

                # bsz x max_bpe_len x hidden_size
                src_outputs = state.encoder_output
                if hasattr(self, 'encoder_mlp') and not only_sc:
                    src_outputs = self.encoder_mlp(src_outputs)

                if first is not None:
                    mask = first.eq(0)
                    src_outputs = src_outputs.gather(
                        index=first.unsqueeze(2).repeat(1, 1,
                                                        src_outputs.size(-1)),
                        dim=1)
                else:
                    mask = state.encoder_mask[:, 51:].eq(0)
                    # src_outputs = self.decoder.embed_tokens(src_tokens)
                mask = mask.unsqueeze(1)
                input_embed = self.decoder.embed_tokens(src_tokens)  # bsz x max_word_len x hidden_size
                input_embed = self.dropout_layer(input_embed)
                if self.avg_feature:  # 先把feature合并一下
                    src_outputs = (src_outputs[:, 51:] + input_embed) / 2
                word_scores = torch.einsum(
                    'blh,bnh->bln', hidden_state,
                    src_outputs[:, 51:])  # bsz x max_len x max_word_len
                if not self.avg_feature:
                    gen_scores = torch.einsum(
                        'blh,bnh->bln', hidden_state,
                        input_embed)  # bsz x max_len x max_word_len
                    word_scores = (gen_scores + word_scores) / 2
                mask = mask.__or__(
                    src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
                word_scores = word_scores.masked_fill(mask, -1e32)
                logits[:, :, self.src_start_index:] = word_scores
                logits[:, :, 1:2] = eos_scores
            return logits
        else:
            bsz, max_len = tokens.size()
            encoder_outputs = state.encoder_output
            encoder_pad_mask = state.encoder_mask

            first = state.first
            # tokens之后的0全是padding，因为1是eos, 在pipe中规定的
            cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
            tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

            # 把输入做一下映射
            mapping_token_mask = tokens.lt(self.src_start_index)  # 为1的地方应该从mapping中取index
            mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
            tag_mapped_tokens = self.mapping[mapped_tokens]
            src_tokens_index = tokens - self.src_start_index  # bsz x num_src_token
            src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
            src_tokens = state.src_tokens
            if first is not None:
                src_tokens = src_tokens.gather(index=first, dim=1)
            word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
            tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
            tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
            if self.training:
                tokens = tokens[:, :-1]
                decoder_pad_mask = tokens.eq(
                    self.pad_token_id)  # decoder需要让pad位置为1
                dict = self.decoder(input_ids=tokens,
                                    encoder_hidden_states=encoder_outputs,
                                    encoder_padding_mask=encoder_pad_mask,
                                    decoder_padding_mask=decoder_pad_mask,
                                    decoder_causal_mask=self.
                                    causal_masks[:tokens.size(1), :tokens.size(1)],
                                    return_dict=True)
            else:
                past_key_values = state.past_key_values
                dict = self.decoder(input_ids=tokens,
                                    encoder_hidden_states=encoder_outputs,
                                    encoder_padding_mask=encoder_pad_mask,
                                    decoder_padding_mask=None,
                                    decoder_causal_mask=self.
                                    causal_masks[:tokens.size(1), :tokens.size(1)],
                                    return_dict=True)
            hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
            hidden_state = self.dropout_layer(hidden_state)
            if not self.training:
                state.past_key_values = dict.past_key_values

            logits = hidden_state.new_full(
                (hidden_state.size(0), hidden_state.size(1),
                 self.src_start_index + src_tokens.size(-1)),
                fill_value=-1e24)
            # 首先计算的是
            if self.need_tag:  # if predict the sentiment or not
                tag_scores = F.linear(
                    hidden_state,
                    self.dropout_layer(
                        self.decoder.embed_tokens.
                        weight[self.label_start_id:self.label_start_id +
                                                   3]))  # bsz x max_len x num_class   相当于在计算向量相似度
                logits[:, :, 3:self.src_start_index] = tag_scores

            if not only_sc:
                eos_scores = F.linear(
                    hidden_state,
                    self.dropout_layer(self.decoder.embed_tokens.weight[2:3]))

                # bsz x max_bpe_len x hidden_size
                src_outputs = state.encoder_output
                if hasattr(self, 'encoder_mlp') and not only_sc:
                    src_outputs = self.encoder_mlp(src_outputs)

                if first is not None:
                    mask = first.eq(0)
                    src_outputs = src_outputs.gather(
                        index=first.unsqueeze(2).repeat(1, 1,
                                                        src_outputs.size(-1)),
                        dim=1)
                else:
                    mask = state.encoder_mask[:, 51:].eq(0)
                    # src_outputs = self.decoder.embed_tokens(src_tokens)
                mask = mask.unsqueeze(1)
                input_embed = self.decoder.embed_tokens(
                    src_tokens)  # bsz x max_word_len x hidden_size
                input_embed = self.dropout_layer(input_embed)
                if self.avg_feature:  # 先把feature合并一下
                    src_outputs = (src_outputs[:, 51:] + input_embed) / 2
                word_scores = torch.einsum(
                    'blh,bnh->bln', hidden_state,
                    src_outputs[:, 51:])  # bsz x max_len x max_word_len
                if not self.avg_feature:
                    gen_scores = torch.einsum(
                        'blh,bnh->bln', hidden_state,
                        input_embed)  # bsz x max_len x max_word_len
                    word_scores = (gen_scores + word_scores) / 2
                mask = mask.__or__(
                    src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
                word_scores = word_scores.masked_fill(mask, -1e32)
                logits[:, :, self.src_start_index:] = word_scores
                logits[:, :, 1:2] = eos_scores

            return logits

    def decode(self, tokens, state,sentiment_value, only_sc=False):
        return self(tokens, state,sentiment_value, only_sc)[:, -1]

class Span_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.loss_fct = nn.CrossEntropyLoss()
        self.fc = nn.LogSoftmax(dim=-1)

    def forward(self, tgt_tokens, pred, mask):

        tgt_tokens = tgt_tokens.masked_fill(mask.eq(0), -100)
        output = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        return output