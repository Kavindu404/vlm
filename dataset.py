import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class VLMDataset(Dataset):

    def __init__(self, image_dir, h5_path, tokenizer, split="train"):

        self.image_dir = image_dir
        self.h5_file = h5py.File(h5_path, 'r')
        self.tokenizer = tokenizer
        self.transform = self._get_default_transforms(split)
        self.split = split
        
        self.max_len = self.h5_file['metadata'].attrs['max_length']
        self.pad_token_id = self.h5_file['metadata'].attrs['pad']
        self.sos_token_id = self.h5_file['metadata'].attrs['startoftext'] 
        self.eos_token_id = self.h5_file['metadata'].attrs['endoftext']
        
        self.image_ids = self.h5_file['metadata']['image_ids'][:]

    def _get_default_transforms(self, split):
        if split == "train":
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        if self.split=="train":
            img_path = f"{self.image_dir}/COCO_train2014_{img_id:012d}.jpg"
        else:
            img_path = f"{self.image_dir}/COCO_val2014_{img_id:012d}.jpg"

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
           
        tokens = self.h5_file[f'captions/{img_id}'][0]  # we only take the first caption for each image
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        return {
            'image': image,
            'tokens': tokens,
            'image_id': img_id
        }

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def __del__(self):
        self.h5_file.close()

class VQADataset(Dataset):

    def __init__(self, tokenizer, h5_path, image_dir="/unity/g1/kgalla/datasets/vqa_dataset/images", split="train"):

        self.image_dir = image_dir
        self.h5_file = h5py.File(h5_path, 'r')
        self.tokenizer = tokenizer
        self.transform = self._get_default_transforms(split)
       
        self.max_question_len = self.h5_file['metadata'].attrs['max_question_length']
        self.max_answer_len = self.h5_file['metadata'].attrs['max_answer_length']
        self.pad_token_id = self.h5_file['metadata'].attrs['pad_token']
        self.sos_token_id = self.h5_file['metadata'].attrs['start_token']
        self.eos_token_id = self.h5_file['metadata'].attrs['eos_token']
        self.question_start_token = self.h5_file['metadata'].attrs['question_start']
        self.question_end_token = self.h5_file['metadata'].attrs['question_end']
        
        self.image_ids = [id.decode() for id in self.h5_file['metadata/image_ids'][:]]

    def _get_default_transforms(self, split):
        if split == "train":
            return T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        if os.path.exists(f"{self.image_dir}/{img_id}.jpg"):
            img_path = f"{self.image_dir}/{img_id}.jpg"
        elif os.path.exists(f"{self.image_dir}/{img_id}.png"):
            img_path = f"{self.image_dir}/{img_id}.png"
        else:
            raise FileNotFoundError(f"No image found for ID {img_id} with jpg or png extension")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        question_tokens = self.h5_file[f'questions/{img_id}'][0]  
        answer_tokens = self.h5_file[f'answers/{img_id}'][0]
        
        question_tokens = torch.tensor(question_tokens, dtype=torch.long)
        answer_tokens = torch.tensor(answer_tokens, dtype=torch.long)
        
        return {
            'image': image,
            'questions': question_tokens,
            'answers': answer_tokens,
            'image_id': img_id
        }

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def __del__(self):
        self.h5_file.close()