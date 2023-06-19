import copy
import torch
import numpy as np
from transformers import BartTokenizer, AutoTokenizer
from itertools import chain
from functools import cmp_to_key
from spacy.lang.en.tag_map import TAG_MAP
import json
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy import displacy


def cmp(v1, v2):
    if v1[0] == v2[0]:
        return v1[1] - v2[1]
    return v1[0] - v2[0]


def matrix_pad(mat, pad_cl):
    # 在pad_cl后面插入0
    assert mat.shape[0] == mat.shape[1]
    mat_len = mat.shape[0]
    indice = list(np.arange(0, mat_len))
    indice.insert(pad_cl + 1, 0)
    mat = mat[:, indice]
    mat[:, pad_cl + 1] = 0
    mat = mat[indice, :]
    mat[pad_cl + 1, :] = 0
    return mat


class ConditionTokenizer:
    """
    tokenizer for image features, event and task type
    this is NOT inherent from transformers Tokenizer
    """

    def __init__(self,
                 args,
                 pretrained_model_name='facebook/bart-base',
                 cls_token="<<cls>>",
                 mlm_token="<<mlm>>",
                 mrm_token="<<mrm>>",
                 trc_token="<<trc>>",
                 begin_text="<<text>>",
                 end_text="<</text>>",
                 img_feat='<<img_feat>>',
                 begin_img="<<img>>",
                 end_img="<</img>>",
                 sc_token='<<SC>>',
                 ae_oe_token="<<AOE>>",
                 sep_token="<<SEP>>",
                 aesc_token='<<AESC>>',
                 pos_token='<<POS>>',
                 neu_token='<<NEU>>',
                 neg_token='<<NEG>>',
                 senti_token='<<senti>>',
                 ANP_token='<<ANP>>',
                 ANP_generate_token='<<AOG>>',
                 ):
        # self._base_tokenizer = BartTokenizer.from_pretrained(
        #     pretrained_model_name, )
        self._base_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, )

        self.additional_special_tokens = [
            cls_token, mlm_token, mrm_token, begin_text, end_text, img_feat,
            begin_img, end_img, senti_token, ANP_token, ANP_generate_token,
            pos_token, neu_token, neg_token, ae_oe_token, sep_token,
            aesc_token, sc_token
        ]
        unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens
        self._base_tokenizer.unique_no_split_tokens = unique_no_split_tokens + self.additional_special_tokens
        self.unique_no_split_tokens = self._base_tokenizer.unique_no_split_tokens

        self._base_tokenizer.add_tokens(self.additional_special_tokens)
        self.cls_token = cls_token
        self.mlm_token = mlm_token
        self.mrm_token = mrm_token
        self.trc_token = trc_token
        self.begin_text = begin_text
        self.end_text = end_text
        self.img_feat = img_feat
        self.begin_img = begin_img
        self.end_img = end_img

        self.sc_token = sc_token
        self.ae_oe_token = ae_oe_token
        self.sep_token = sep_token
        self.senti_token = senti_token
        self.ANP_token = ANP_token
        self.ANP_generate_token = ANP_generate_token

        self.aesc_token = aesc_token
        self.pos_token = pos_token
        self.neu_token = neu_token
        self.neg_token = neg_token

        self.cls_token_id = self.convert_tokens_to_ids(cls_token)
        self.mlm_token_id = self.convert_tokens_to_ids(mlm_token)
        self.mrm_token_id = self.convert_tokens_to_ids(mrm_token)
        self.trc_token_id = self.convert_tokens_to_ids(trc_token)
        self.begin_text_id = self.convert_tokens_to_ids(begin_text)
        self.end_text_id = self.convert_tokens_to_ids(end_text)
        self.img_feat_id = self.convert_tokens_to_ids(img_feat)
        self.begin_img_id = self.convert_tokens_to_ids(begin_img)
        self.end_img_id = self.convert_tokens_to_ids(end_img)

        self.sc_token_id = self.convert_tokens_to_ids(sc_token)
        self.ae_oe_token_id = self.convert_tokens_to_ids(ae_oe_token)
        self.sep_token_id = self.convert_tokens_to_ids(sep_token)
        self.senti_token_id = self.convert_tokens_to_ids(senti_token)
        self.ANP_token_id = self.convert_tokens_to_ids(ANP_token)
        self.ANP_generate_token_id = self.convert_tokens_to_ids(
            ANP_generate_token)
        self.aesc_token_id = self.convert_tokens_to_ids(aesc_token)
        self.pos_token_id = self.convert_tokens_to_ids(pos_token)
        self.neu_token_id = self.convert_tokens_to_ids(neu_token)
        self.neg_token_id = self.convert_tokens_to_ids(neg_token)

        self.vocab_size = self._base_tokenizer.vocab_size
        self.bos_token = self._base_tokenizer.bos_token
        self.bos_token_id = self._base_tokenizer.bos_token_id

        self.eos_token = self._base_tokenizer.eos_token
        self.eos_token_id = self._base_tokenizer.eos_token_id
        self.pad_token = self._base_tokenizer.pad_token
        self.pad_token_id = self._base_tokenizer.pad_token_id
        self.unk_token = self._base_tokenizer.unk_token
        self.unk_token_id = self._base_tokenizer.unk_token_id
        self.sentinet_on = args.sentinet_on
        self.gcn_on = args.gcn_on

        if self.sentinet_on:
            path = '/home/zhouru/ABSA4/src/senticnet_word.txt'
            self.senticNet = {}
            fp = open(path, 'r')
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                word, sentic = line.split('\t')
                self.senticNet[word] = sentic
            fp.close()

        print('self.bos_token_id', self.bos_token_id)
        print('self.eos_token_id', self.eos_token_id)
        print('self.pad_token_id', self.pad_token_id)
        if args.task == 'pretrain':
            self.mapping = {'AE_OE': '<<AOE>>', 'SEP': '<<SEP>>'}
        else:
            self.mapping = {
                'AESC': '<<AESC>>',
                'POS': '<<POS>>',
                'NEU': '<<NEU>>',
                'NEG': '<<NEG>>'
            }
        self.senti = {'POS': '<<POS>>', 'NEU': '<<NEU>>', 'NEG': '<<NEG>>'}
        self.senti2id = {}
        for key, value in self.senti.items():
            key_id = self._base_tokenizer.convert_tokens_to_ids(
                self._base_tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            # assert key_id[0] >= self.cur_num_tokens
            self.senti2id[key] = key_id[0]
        self.mapping2id = {}
        self.mapping2targetid = {}
        for key, value in self.mapping.items():
            key_id = self._base_tokenizer.convert_tokens_to_ids(
                self._base_tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            # assert key_id[0] >= self.cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid) + 2
        print(self.mapping2id)

    def encode(self, *args, **kwargs):
        return self._base_tokenizer(*args, **kwargs)

    def pad_tokens(self, tokens, noun_masks=None, dependency_matrix=None):
        max_len = max([len(x) for x in tokens])

        pad_result = torch.full((len(tokens), max_len), self.pad_token_id)
        mask = torch.zeros(pad_result.size(), dtype=torch.bool)
        for i, x in enumerate(tokens):
            pad_result[i, :len(x)] = torch.tensor(tokens[i], dtype=torch.long)
            mask[i, :len(x)] = True

        if noun_masks is not None:
            noun_mask = torch.zeros((len(tokens), max_len), dtype=torch.bool)
            for i, x in enumerate(tokens):
                noun_mask[i, :len(x)] = torch.tensor(noun_masks[i], dtype=torch.bool)

        if dependency_matrix is not None:
            ret_dependency_matrix = torch.zeros([len(tokens), max_len, max_len], dtype=torch.float)
            for i in range(len(tokens)):
                dim = dependency_matrix[i].shape[0]
                ret_dependency_matrix[i, :dim, :dim] = dependency_matrix[i]

        if noun_masks is None:
            noun_mask = None
        if dependency_matrix is None:
            ret_dependency_matrix = None

        return pad_result, mask, noun_mask, ret_dependency_matrix

    def encode_condition(self, img_num=None, sentence=None, text_only=False):
        """
        tokenize text, image features and event
        the output format (after decoded back):
        task_type [<img> <img_feat> ... <img_feat> </img>] [<event> EVENT </event>] [<mlm> MLM </mlm>]

        :param task_type: str or list[str]
        :param img_num: int or list[int], the number of image features
        :param event: str or list[str], event descriptions
        :param mlm: str or list[str], sentence for masked language modeling
        :return: dict {str: Tensor}, {
                "input_ids": ...,
                "attention_mask": ...,
                "event_mask": ...,          only exist if event is given. 1 for the position with event tokens
                "mlm_mask": ...,            only exist if mlm is given. 1 for the position with mlm tokens
                "img_mask":...,             only exist if img_num is given. 1 for the position with img tokens
            }
        """

        image_text = None
        if img_num is not None:
            if not isinstance(img_num, list):
                img_num = [img_num]
            image_text = []
            for index, value in enumerate(img_num):
                image_text.append(self.begin_img + self.img_feat * value +
                                  self.end_img)
        if sentence is not None:
            noun_list = ['NNP', 'NNPS', 'NN', 'NNS']
            if not isinstance(sentence, list):
                sentence = [sentence]
            sentence_split = [x.split() for x in sentence]
            pos_doc = [nlp(x) for x in sentence]

            noun_positions = []

            for i, split in enumerate(sentence_split):
                if len(sentence_split[i]) != len(pos_doc[i]):
                    new_sentence = []
                    for token in pos_doc[i]:
                        new_sentence.append(str(token))
                    sentence_split[i] = new_sentence
                    assert len(sentence_split[i]) == len(pos_doc[i])
                noun_position = []
                for j in range(len(pos_doc[i])):
                    if pos_doc[i][j].tag_ in noun_list:
                        noun_position.append(j)
                noun_positions.append(noun_position)

            # 处理依赖矩阵
            if self.gcn_on:
                dependency_matrix = [torch.zeros([len(sentence_split[i]), len(sentence_split[i])]) for i in
                                 range(len(sentence_split))]
                for i, split in enumerate(sentence_split):
                    # assert len(sentence_split[i]) == len(pos_doc[i])
                    for t in pos_doc[i]:
                        dependency_matrix[i][t.i][t.i] = 5
                        for child in t.children:
                            dependency_matrix[i][t.i][child.i] = 1
                            dependency_matrix[i][child.i][t.i] = 1

                            for cchild in child.children:
                                dependency_matrix[i][t.i][cchild.i] = 1
                                dependency_matrix[i][cchild.i][t.i] = 1
            else:
                dependency_matrix=None

            input_sentence_tokens = []
            assert len(sentence_split) == len(pos_doc)
            noun_masks = []
            # token_index 保存每个句子中每个token在 dependency_matrix对应的起始位置

            # right
            # token_index=[np.arange(1,len(x)+1) for x in sentence_split]

            token_index = [np.arange(0, len(x)) for x in sentence_split]
            sentiments = []
            for i, split in enumerate(sentence_split):
                noun_mask = [0]
                word_bpes = [self.bos_token_id]
                if self.gcn_on:
                    # # 扩展依赖矩阵
                    dependency_matrix[i] = matrix_pad(dependency_matrix[i], -1)
                    dependency_matrix[i][0][0] = 1
                sentiment = [0]
                for j, word in enumerate(split):
                    bpes = self._base_tokenizer.tokenize(word, add_prefix_space=True)
                    bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                    if j in noun_positions[i]:
                        noun_mask += [1] * len(bpes)
                    else:
                        noun_mask += [0] * len(bpes)

                    if self.sentinet_on:
                        if word in self.senticNet:
                            sentiment.extend(len(bpes) * [float(self.senticNet[word])])
                        else:
                            # sentiment.extend(len(bpes)*[0])
                            for s in bpes:
                                if s in self.senticNet:
                                    sentiment.append(float(self.senticNet[s]))
                                else:
                                    sentiment.append(0)
                    if self.gcn_on:
                        # 扩展依赖矩阵
                        if len(bpes) > 1:
                            # 依赖矩阵扩展len(bpes)-1行
                            for d_i in range(len(bpes) - 1):
                                # 在pad_index后插入
                                pad_index = token_index[i][j] + d_i
                                dependency_matrix[i] = matrix_pad(dependency_matrix[i], pad_index)
                                dependency_matrix[i][pad_index + 1][pad_index + 1] = 5
                                # 找j行不为0的部分
                                have_arc = torch.nonzero(
                                    dependency_matrix[i][token_index[i][j]] == 1).squeeze().numpy().tolist()
                                if isinstance(have_arc, int):
                                    have_arc = [have_arc]
                                for arc_x in have_arc:
                                    if arc_x != token_index[i][j]:
                                        dependency_matrix[i][pad_index + 1][arc_x] = 1
                                        dependency_matrix[i][arc_x][pad_index + 1] = 1
                            # token_index该token后位置调整
                            for d_j in range(j + 1, len(split)):
                                token_index[i][d_j] += len(bpes)
                                token_index[i][d_j] -= 1
                    word_bpes.extend(bpes)
                word_bpes.append(self.eos_token_id)
                if self.sentinet_on:
                    sentiment.append(0)
                    sentiments.append(sentiment)
                    # assert len(word_bpes) == len(sentiment)

                # # 扩展依赖矩阵
                if self.gcn_on:
                    dependency_matrix[i] = matrix_pad(dependency_matrix[i], dependency_matrix[i].shape[0] - 1)
                    dependency_matrix[i][-1][-1] = 1
                # assert len(word_bpes)==dependency_matrix[i].shape[0]
                noun_mask += [0]
                # _word_bpes = list(chain(*word_bpes))
                # input_sentence_tokens.append(_word_bpes.copy())
                input_sentence_tokens.append(word_bpes)
                # assert len(word_bpes)==len(noun_mask)
                noun_masks.append(noun_mask)
            # assert len(input_sentence_tokens)==len(noun_masks)

        encoded = {}
        if image_text is not None:
            image_sentence = self.encode(image_text,
                                         add_special_tokens=False,
                                         return_tensors='pt',
                                         padding=True)
            image_ids = image_sentence['input_ids']
            image_attention_mask = image_sentence['attention_mask']
            # input_sentence_tokens, input_sentence_mask, noun_mask= self.pad_tokens(
            #     input_sentence_tokens, noun_masks)

            input_sentence_tokens, input_sentence_mask, noun_mask, dependency_matrix = self.pad_tokens(
                input_sentence_tokens, noun_masks, dependency_matrix)

            # 填充情感值
            if self.sentinet_on:
                sentiment_value = torch.zeros([input_sentence_tokens.shape[0], input_sentence_tokens.shape[1]],
                                              dtype=torch.float)
                for i, x in enumerate(sentiments):
                    sentiment_value[i, :len(x)] = torch.tensor(x, dtype=torch.float)

            if text_only:
                image_attention_mask = torch.zeros(image_ids.size())
            input_ids = torch.cat((image_ids, input_sentence_tokens), 1)
            attention_mask = torch.cat(
                (image_attention_mask, input_sentence_mask), 1)
            noun_mask = torch.cat((torch.zeros(image_ids.size(), dtype=torch.bool), noun_mask), 1)
            # assert attention_mask.shape==noun_mask.shape
        else:
            input_sentence_tokens, input_sentence_mask, noun_mask, dependency_matrix, _ = self.pad_tokens(
                input_sentence_tokens, noun_masks, dependency_matrix)
            input_ids = input_sentence_tokens
            attention_mask = input_sentence_mask

        encoded['input_ids'] = input_ids

        encoded['attention_mask'] = attention_mask
        encoded['noun_mask'] = noun_mask
        encoded['dependency_matrix'] = dependency_matrix
        if self.sentinet_on == False:
            sentiment_value = None
        encoded['sentiment_value'] = sentiment_value
        # build mlm mask
        if sentence is not None:
            sentence_mask = torch.zeros(input_ids.size(), dtype=torch.bool)
            for index, value in enumerate(input_ids):
                start = (value == self.bos_token_id).nonzero(as_tuple=True)[0]
                end = (value == self.eos_token_id).nonzero(as_tuple=True)[0]
                sentence_mask[index, start + 1:end] = True
            encoded['sentence_mask'] = sentence_mask

        # build img mask
        if img_num is not None:
            encoded['img_mask'] = encoded['input_ids'] == self.img_feat_id

        return encoded


    def encode_label(self, label, img_num=None):  # generate labels for MLM task

        # build text label
        if not isinstance(label, list):
            label = [label]

        label_split = [x.split() for x in label]
        label_tokens = []
        for split in label_split:
            word_bpes = [[self.bos_token_id], [self.mlm_token_id]]
            for word in split:
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.eos_token_id])
            _word_bpes = list(chain(*word_bpes))
            label_tokens.append(_word_bpes)
        input_ids, attention_mask = self.pad_tokens(label_tokens)

        output_shape = input_ids[:, 2:].shape
        labels = torch.empty(output_shape, dtype=torch.long)
        decoder_input_ids = torch.empty(input_ids[:, 1:].shape,
                                        dtype=torch.long)
        decoder_attention_mask = torch.empty(input_ids[:, 1:].shape,
                                             dtype=torch.long)

        for i in range(labels.size(0)):
            labels[i] = input_ids[i][(input_ids[i] != self.bos_token_id)
                                     & (input_ids[i] != self.mlm_token_id)]
            decoder_input_ids[i] = input_ids[i][
                input_ids[i] != self.eos_token_id]
            decoder_attention_mask[i] = attention_mask[i][
                input_ids[i] != self.eos_token_id]
        labels[(labels == self.pad_token_id) | (labels == self.begin_img_id) |
               (labels == self.end_img_id) | (labels == self.mlm_token_id) |
               (labels == self.img_feat_id)] = -100
        output = {
            'mlm_labels': labels,
            'mlm_decoder_input_ids': decoder_input_ids,
            'mlm_decoder_attention_mask': decoder_attention_mask
        }

        return output

    def encode_aesc(self, label, aesc_spans, aesc_max_len):
        target_shift = len(self.mapping2targetid) + 2
        # print(target_shift)
        aesc_text = []
        masks = []
        gt_spans = []

        flag = True
        for text, span in zip(label, aesc_spans):
            span = sorted(span, key=cmp_to_key(cmp))
            word_bpes = [[self.begin_text_id]]
            for word in text.split():
                bpes = self._base_tokenizer.tokenize(word,
                                                     add_prefix_space=True)
                bpes = self._base_tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.end_text_id])
            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            cur_text = [
                0, self.mapping2targetid['AESC'], self.mapping2targetid['AESC']
            ]
            mask = [0, 0, 0]
            gt = []
            for x in span:
                s_bpe = cum_lens[x[0]] + target_shift
                e_bpe = cum_lens[x[1] - 1] + target_shift
                polarity = self.mapping2targetid[x[2]]
                cur_text.append(s_bpe)
                cur_text.append(e_bpe)
                cur_text.append(polarity)
                gt.append((s_bpe, e_bpe, polarity))
                mask.append(1)
                mask.append(1)
                mask.append(1)
            cur_text.append(1)
            mask.append(1)
            aesc_text.append(cur_text)
            gt_spans.append(gt)
            masks.append(mask)
        span_max_len = max([len(x) for x in aesc_text])
        for i in range(len(masks)):
            add_len = span_max_len - len(masks[i])
            masks[i] = masks[i] + [0 for ss in range(add_len)]
            aesc_text[i] = aesc_text[i] + [1 for ss in range(add_len)]

        output = {}
        output['labels'] = torch.tensor(aesc_text)
        output['masks'] = torch.tensor(masks)
        output['spans'] = gt_spans
        return output

    def decode(self, token_ids, skip_special_tokens=False):
        return self._base_tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, tokens):
        return self._base_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids):
        return self._base_tokenizer.convert_ids_to_tokens(ids)

    def get_base_tokenizer(self):
        return self._base_tokenizer

    def __len__(self):
        return len(self._base_tokenizer)
