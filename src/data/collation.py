import warnings
import numpy as np
import torch
from itertools import chain

class Collator:
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """
    def __init__(self,
                 tokenizer,
                 has_label=True,
                 aesc_enabled=False,
                 text_only=False,
                 trc_enabled=False,
                 lm_max_len=30,
                 max_img_num=49,
                 max_span_len=20):
        """
        :param tokenizer: ConditionTokenizer
        :param mlm_enabled: bool, if use mlm for language modeling. False for autoregressive modeling
        :param mrm_enabled: bool, if use mrm
        :param rp_enabled: bool, if use relation prediction (VG)
        :param ap_enabled: bool, if use attribute prediction (VG)
        :param mlm_probability: float, probability to mask the tokens
        :param mrm_probability: float, probability to mask the regions
        """
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._aesc_enabled = aesc_enabled
        self._trc_enabled=trc_enabled
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._max_span_len = max_span_len
        self.text_only = text_only
        if not has_label:
            raise ValueError(
                'mlm_enabled can not be true while has_label is false. MLM need labels.'
            )

    def _clip_text(self, text, length):
        tokenized = []
        for i, word in enumerate(text.split()):
            if i == 0:
                bpes = self._tokenizer._base_tokenizer.tokenize(word)
            else:
                bpes = self._tokenizer._base_tokenizer.tokenize(
                    word, add_prefix_space=True)
            bpes = self._tokenizer._base_tokenizer.convert_tokens_to_ids(bpes)
            tokenized.append(bpes)
        _tokenized = list(chain(*tokenized))
        return self._tokenizer.get_base_tokenizer().decode(_tokenized[:length])

    def __call__(self, batch):
        batch = [entry for entry in batch if entry is not None]
        image_features =[x['img_feat'] for x in batch]

        img_num = [49]*len(image_features)

        target = [x['sentence'] for x in batch]
        sentence = list(target)

        encoded_conditions = self._tokenizer.encode_condition(
            img_num=img_num, sentence=sentence, text_only=self.text_only)

        input_ids = encoded_conditions['input_ids']
        output = {}
        condition_img_mask = encoded_conditions['img_mask']

        output['input_ids'] = input_ids
        output['attention_mask'] = encoded_conditions['attention_mask']
        output['image_features'] = image_features
        output['input_ids'] = input_ids
        output['sentiment_value']=encoded_conditions['sentiment_value']

        output['noun_mask']=encoded_conditions['noun_mask']
        output['dependency_matrix']=encoded_conditions['dependency_matrix']



        if self._has_label:
            if self._aesc_enabled:
                output['AESC'] = self._tokenizer.encode_aesc(
                    target, [x['aesc_spans'] for x in batch],  # target = [x['sentence'] for x in batch]
                    self._max_span_len)
                output['task'] = 'AESC'
            if self._trc_enabled:
                output['ifpairs']=[x['ifpairs'] for x in batch]

        output['image_id'] = [x['image_id'] for x in batch]
        if self._trc_enabled==False:
            output['gt'] = [x['gt'] for x in batch]
        return output