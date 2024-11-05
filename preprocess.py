import json
import h5py
import tiktoken
import numpy as np
from tqdm import tqdm
from pathlib import Path



def get_tokenizer():

    base_tokenizer = tiktoken.get_encoding("gpt2")
    
    tokenizer = tiktoken.Encoding(
        name="gpt2_with_special_tokens",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks= base_tokenizer._mergeable_ranks,
        special_tokens= {
            **base_tokenizer._special_tokens,
            "<|startoftext|>": 50257,
            "<|pad|>": 50258
        }
    )

    return tokenizer

def prepare_dataset(tokenizer, annotation_path, output_path, max_len=196):

    with open(annotation_path, 'r') as f:
        data = json.load(f)['annotations']

    with h5py.File(output_path, 'w') as h5f:
        captions_group = h5f.create_group('captions')
        metadata = h5f.create_group('metadata')

        metadata.attrs['max_length'] = max_len
        metadata.attrs['endoftext'] = 50256
        metadata.attrs['startoftext'] = 50257
        metadata.attrs['pad'] = 50258

        image_captions = {}

        for item in tqdm(data):

            image_id = item['image_id']
            caption = item['caption']
            
            tokens = tokenizer.encode(caption)
            tokens = [50257] + tokens + [50256]

            if len(tokens)<max_len:
                tokens.extend([50258]*(max_len-len(tokens)))
            else:
                print(f"The caption is too long. The max_len is {max_len} but the caption is {len(tokens)}")
            
            if image_id not in image_captions:
                image_captions[image_id] = []
            image_captions[image_id].append(tokens)

        for image_id, tokens_list in tqdm(image_captions.items(), desc="Saving to H5"):
            image_tokens = np.array(tokens_list, dtype=np.int32)
            captions_group.create_dataset(
                str(image_id),
                data=image_tokens,
                compression='gzip',
                compression_opts=9
            )
        
        metadata.create_dataset(
            'image_ids',
            data=np.array(list(image_captions.keys()), dtype=np.int32)
        )

    


if __name__ == "__main__":
    
    tokenizer = get_tokenizer()

    train_annotation_path = "/unity/g1/kgalla/datasets/annotations/captions_train2014.json"
    train_output_path = "/unity/g1/kgalla/datasets/annotations/coco_train2014_captions.h5"

    val_annotation_path = "/unity/g1/kgalla/datasets/annotations/captions_val2014.json"
    val_output_path = "/unity/g1/kgalla/datasets/annotations/coco_val2014_captions.h5"

    prepare_dataset(tokenizer, train_annotation_path, train_output_path)
    prepare_dataset(tokenizer, val_annotation_path, val_output_path)