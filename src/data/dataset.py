import torch
import numpy as np
import json
import csv
import os
import json
import torch.utils.data as data
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import pdb

class Twitter_Dataset(data.Dataset):
    def __init__(self,img_path,infos, split):
        self.path_img = img_path
        self.infos = json.load(open(infos, 'r'))

        if split == 'train':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/train.json', 'r'))
        elif split == 'dev':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/dev.json', 'r'))
        elif split == 'test':
            self.data_set = json.load(
                open(self.infos['data_dir'] + '/test.json', 'r'))
        else:
            raise RuntimeError("split type is not exist!!!")

        crop_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])


        # crop_size=224
        # self.transform = transforms.Compose([
        #     transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406),
        #                          (0.229, 0.224, 0.225))])
        self.count_img_error=0

    def __len__(self):
        return len(self.data_set)

    def get_img_feature(self,id):
        image_path = os.path.join(self.path_img, id)
        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except:
            self.count_img_error += 1
            # print('image has problem!')
            image_path_fail = os.path.join(self.path_img, '17_06_4705.jpg')
            image = Image.open(image_path_fail).convert('RGB')
            image = self.transform(image)
        return image


    def get_aesc_spans(self, dic):
        aesc_spans = []
        for x in dic:
            aesc_spans.append((x['from'], x['to'], x['polarity']))
        return aesc_spans

    def get_gt_aspect_senti(self, dic):
        gt = []
        for x in dic:
            gt.append((' '.join(x['term']), x['polarity']))
        return gt

    def __getitem__(self, index):
        output = {}
        data = self.data_set[index]
        img_id = data['image_id']
        img_feature = self.get_img_feature(img_id)
        output['img_feat'] = img_feature

        output['sentence'] = ' '.join(data['words'])
        # add
        output['noun']=data['noun']

        aesc_spans = self.get_aesc_spans(data['aspects'])
        output['aesc_spans'] = aesc_spans
        output['image_id'] = img_id
        gt = self.get_gt_aspect_senti(data['aspects'])
        output['gt'] = gt
        return output


class TRC_Dataset(data.Dataset):
    def __init__(self,infos):
        self.infos = json.load(open(infos, 'r'))
        self.path_img = self.infos["img_path"]
        self.data_set = json.load(open(self.infos['data_dir'] + '/trc.json', 'r'))

        crop_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),  # args.crop_size, by default it is set to be 224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        self.count_img_error=0

    def __len__(self):
        return len(self.data_set)


    def get_img_feature(self,id):
        image_path = os.path.join(self.path_img, id)
        if not os.path.exists(image_path):
            print(image_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        # try:
        #     image = Image.open(image_path).convert('RGB')
        #     image = self.transform(image)
        # except:
        #     with open('/home/zhouru/ABSA3/image_not_found.txt','a') as file:
        #         file.write(id+'\r')
        #     self.count_img_error += 1
        #     # print('image has problem!')
        #     image_path_fail = os.path.join('/home/zhouru/IJCAI2019_data/twitter2015_images', '17_06_4705.jpg')
        #     image = Image.open(image_path_fail).convert('RGB')
        #     image = self.transform(image)
            # print("img_not_found={}".format(self.count_img_error))
        return image


    def __getitem__(self, index):
        output = {}
        data = self.data_set[index]
        img_id = data['image_id']
        img_feature = self.get_img_feature(img_id)
        output['img_feat'] = img_feature
        output['sentence'] = ' '.join(data['words'])
        output['image_id'] = img_id
        output['ifpairs'] = data['ifpairs'][1]
        return output