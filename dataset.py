import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

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