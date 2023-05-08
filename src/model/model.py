from typing import Optional, Tuple
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch
import torch.nn.functional as F
from torch import nn
from src.model.modeling_bart import (PretrainedBartModel, BartEncoder,
                                     BartDecoder, BartModel,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)

from transformers import BartTokenizer

from src.model.config import MultiModalBartConfig
from src.model.mixins import GenerationMixin, FromPretrainedMixin
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span, Span_loss
from src.model.MAESC_model import Attention
import numpy as np
from src.model.GAT import GAT
from src.model.GCN import GCN



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new

            
class TRCPretrain(FromPretrainedMixin, PretrainedBartModel):
    def build_model(self,
                    args,
                    bart_model,
                    tokenizer,
                    label_ids,
                    config,
                    decoder_type=None,
                    copy_gate=False,
                    use_encoder_mlp=False,
                    use_recur_pos=False,
                    tag_first=False):
        if args.bart_init:
            model = BartModel.from_pretrained(bart_model)
            num_tokens, _ = model.encoder.embed_tokens.weight.shape
            print('num_tokens', num_tokens)

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            # if use_recur_pos:
            #     decoder.set_position_embedding(label_ids[0], tag_first)

            _tokenizer = BartTokenizer.from_pretrained(bart_model)

            for token in tokenizer.unique_no_split_tokens:
                if token[:2] == '<<':  # 特殊字符
                    index = tokenizer.convert_tokens_to_ids(
                        tokenizer._base_tokenizer.tokenize(token))
                    if len(index) > 1:
                        raise RuntimeError(f"{token} wrong split")
                    else:
                        index = index[0]
                    assert index >= num_tokens, (index, num_tokens, token)
                    indexes = _tokenizer.convert_tokens_to_ids(
                        _tokenizer.tokenize(token[2:-2]))
                    embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                    for i in indexes[1:]:
                        embed += model.decoder.embed_tokens.weight.data[i]
                    embed /= len(indexes)
                    model.decoder.embed_tokens.weight.data[index] = embed
        else:
            raise RuntimeError("error init!!!!!!!")

        multimodal_encoder = MultiModalBartEncoder(config, encoder,
                                                   tokenizer.img_feat_id,
                                                   tokenizer.cls_token_id)
        return (multimodal_encoder, decoder)

    def __init__(self, config: MultiModalBartConfig, bart_model, tokenizer,
                 label_ids, senti_ids, args):
        super().__init__(config)
        self.config = config
        self.mydevice=args.device
        label_ids = sorted(label_ids)
        multimodal_encoder, share_decoder = self.build_model(
            args, bart_model, tokenizer, label_ids, config)
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.encoder = multimodal_encoder
        only_sc = False
        need_tag = True  #if predict the sentiment or not

        self.decoder = MultiModalBartDecoder_span(self.config,
                                                  tokenizer,
                                                  share_decoder,
                                                  tokenizer.pad_token_id,
                                                  label_ids,
                                                  self.causal_mask,
                                                  args.gcn_on,
                                                  need_tag=need_tag,
                                                  only_sc=False)
        self.span_loss_fct = Span_loss()

        # add
        self.noun_linear=nn.Linear(768,768)
        self.multi_linear=nn.Linear(768,768)
        self.att_linear=nn.Linear(768*2,1)
        self.attention=Attention(4,768,768)
        self.linear=nn.Linear(768*2,1)
        self.linear2=nn.Linear(768*2,1)

        self.alpha_linear1=nn.Linear(768,768)
        self.alpha_linear2=nn.Linear(768,768)

        self.trc_classification=nn.Linear(768,2)
        self.softmax=nn.Softmax(dim=-1)




    def get_noun_embed(self,feature,noun_mask):
        # print(feature.shape,noun_mask.shape)
        noun_mask = noun_mask.cpu()
        noun_num = [x.numpy().tolist().count(1) for x in noun_mask]
        noun_position=[np.where(np.array(x)==1)[0].tolist() for x in noun_mask]
        for i,x in enumerate(noun_position):
            assert len(x)==noun_num[i]
        max_noun_num = max(noun_num)

        # pad
        for i,x in enumerate(noun_position):
            if len(x)<max_noun_num:
                noun_position[i]+=[0]*(max_noun_num-len(x))
        noun_position=torch.tensor(noun_position).to(self.mydevice)
        noun_embed=torch.zeros(feature.shape[0],max_noun_num,feature.shape[-1]).to(self.mydevice)
        for i in range(len(feature)):
            noun_embed[i]=torch.index_select(feature[i],dim=0,index=noun_position[i])
            noun_embed[i,noun_num[i]:]=torch.zeros(max_noun_num-noun_num[i],feature.shape[-1])
        # print(noun_embed.shape)
        # pdb.set_trace()
        return noun_embed

    def prepare_state(self,
                      input_ids,
                      image_features,
                      # noun_ids,
                      noun_mask,
                      attention_mask=None,
                      first=None):
        dict = self.encoder(input_ids=input_ids,
                            image_features=image_features,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        src_embed_outputs = hidden_states[0]

        # add
        # 获取名词的embedding
        noun_embed=self.get_noun_embed(encoder_outputs,noun_mask)
        att_features=self.noun_attention(encoder_outputs,noun_embed,mode='cat')
        return att_features


    def noun_attention(self,encoder_outputs,noun_embed,mode='multi-head'):
        if mode=='cat':
            multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
            noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
            noun_features_rep = self.noun_linear(noun_features_rep)
            multi_features_rep = self.multi_linear(multi_features_rep)
            concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
            att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
            att_features = torch.matmul(att, noun_embed)

            alpha = torch.sigmoid(self.linear(torch.cat([self.alpha_linear1(encoder_outputs), self.alpha_linear2(att_features)], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)

            return att_features
        elif mode=='none':
            return encoder_outputs
        elif mode=='multi-head':
            # 多头注意力
            att_features=self.attention(encoder_outputs,noun_embed,noun_embed)
            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(1 - alpha, encoder_outputs) + torch.mul(alpha, att_features)
            return encoder_outputs
        elif mode=='cos_':
            multi_features_rep = encoder_outputs.unsqueeze(1).repeat(1, noun_embed.shape[1], 1,1)
            noun_features_rep = noun_embed.unsqueeze(2).repeat(1, 1,encoder_outputs.shape[1], 1)
            att=torch.cosine_similarity(multi_features_rep,noun_features_rep,dim=-1)
            att=att.max(1)[1]
            att_features=torch.zeros(encoder_outputs.shape).to(self.mydevice)
            for i in range(noun_embed.shape[0]):
                att_features[i]=torch.index_select(noun_embed[i],0,att[i])
            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(alpha, encoder_outputs) + torch.mul(1-alpha, att_features)
            return encoder_outputs

    def forward(
            self,
            input_ids,
            image_features,
            noun_mask,
            attention_mask=None,
            ifpairs=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        att_features = self.prepare_state(input_ids, image_features,noun_mask, attention_mask)
        image_patch_features=att_features[:,1:50,:]
        image_CLS=att_features[0]
        average_img=image_patch_features.mean(-2)
        cls=self.softmax(self.trc_classification(average_img))
        return cls


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs,mix_feature):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs
        self.mix_feature=mix_feature

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)

        if self.mix_feature is not None:
            self.mix_feature = self._reorder_state(self.mix_feature,
                                                         indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new