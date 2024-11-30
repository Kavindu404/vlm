import json
import h5py
import os
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from functools import partial



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

def get_vqa_tokenizer():

    base_tokenizer = tiktoken.get_encoding("gpt2")
    
    tokenizer = tiktoken.Encoding(
        name="gpt2_with_special_tokens",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks= base_tokenizer._mergeable_ranks,
        special_tokens= {
            **base_tokenizer._special_tokens,
            "<|startoftext|>": 50257,
            "<|pad|>": 50258,
            "<|startofquestion|>": 50259,
            "<|endofquestion|>": 50260
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

def save_qa_batch(batch_data, output_path, lock):
    """Process a batch of image QA pairs"""
    with lock:
        with h5py.File(output_path, 'a') as h5f:
            for image_id, qa_data in batch_data:
                questions_array = np.array(qa_data['questions'], dtype=np.int32)
                answers_array = np.array(qa_data['answers'], dtype=np.int32)
                
                if f'questions/{image_id}' not in h5f:
                    h5f['questions'].create_dataset(
                        str(image_id),
                        data=questions_array,
                        compression='gzip',
                        compression_opts=9
                    )
                    
                    h5f['answers'].create_dataset(
                        str(image_id),
                        data=answers_array,
                        compression='gzip',
                        compression_opts=9
                    )

def prepare_gqa_dataset(tokenizer, questions_dir, split, output_path, max_question_len=48, max_answer_len=48):
    image_qa_pairs = {}
    
    # Process questions
    if split == 'train':
        for i in range(1):
            file_path = os.path.join(questions_dir, f'train_all_questions_{i}.json')
            print(f"\nProcessing {file_path}")
            
            with open(file_path, 'r') as f:
                questions_data = json.load(f)
                
            image_qa_pairs, skipped = process_questions(questions_data, image_qa_pairs, tokenizer, 
                                                      max_question_len, max_answer_len)
    
    elif split == 'val':
        file_path = os.path.join(questions_dir, 'val_all_questions.json')
        print(f"\nProcessing {file_path}")
        
        with open(file_path, 'r') as f:
            questions_data = json.load(f)
            
        image_qa_pairs, skipped = process_questions(questions_data, image_qa_pairs, tokenizer, 
                                                  max_question_len, max_answer_len)
    
    else:
        raise ValueError("Split must be either 'train' or 'val'")

    # Initialize H5 file with structure first
    with h5py.File(output_path, 'w') as h5f:
        h5f.create_group('questions')
        h5f.create_group('answers')
        metadata = h5f.create_group('metadata')

        # Set metadata attributes
        metadata.attrs['max_question_length'] = max_question_len
        metadata.attrs['max_answer_length'] = max_answer_len
        metadata.attrs['pad_token'] = 50258
        metadata.attrs['start_token'] = 50257
        metadata.attrs['eos_token'] = 50256
        metadata.attrs['question_start'] = 50259
        metadata.attrs['question_end'] = 50260
        metadata.attrs['num_image_tokens'] = 196
        metadata.attrs['total_sequence_length'] = 196 + max_question_len
        
        # Save image IDs
        metadata.create_dataset(
            'image_ids',
            data=np.array(list(image_qa_pairs.keys()), dtype='S')
        )
        
        # Save image paths
        for image_id, qa_data in image_qa_pairs.items():
            metadata.attrs[f'image_path_{image_id}'] = qa_data['image_path'].encode('utf-8')

    # Prepare data for parallel processing
    items = list(image_qa_pairs.items())
    num_cores = min(mp.cpu_count(), 32)  # Limit cores if needed
    batch_size = max(1, len(items) // (num_cores * 4))
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    print(f"Processing {len(batches)} batches using {num_cores} cores")
    
    # Create a lock for H5 file access
    lock = mp.Manager().Lock()
    
    # Create pool and process batches
    with mp.Pool(num_cores) as pool:
        save_func = partial(save_qa_batch, output_path=output_path, lock=lock)
        list(tqdm(pool.imap(save_func, batches), total=len(batches), desc="Saving to H5"))

    print("\nDataset preparation completed!")
    print(f"Total images processed: {len(image_qa_pairs)}")
    print(f"Total QA pairs skipped: {skipped}")

    return image_qa_pairs

def process_questions(questions_data, image_qa_pairs, tokenizer, max_question_len, max_answer_len):
    """Helper function to process questions and build image_qa_pairs"""
    skipped = 0
    for question_id, question_info in tqdm(questions_data.items()):
        image_id = question_info['imageId']
        question = question_info['question']
        answer = question_info['fullAnswer']
        
        # Encode question
        q_tokens = [50259]  # <|startofquestion|>
        q_tokens += tokenizer.encode(question)
        q_tokens += [50260]  # <|endofquestion|>
        
        # Encode answer
        a_tokens = [50257]  # <|startoftext|>
        a_tokens += tokenizer.encode(answer)
        a_tokens += [50256]  # <|endoftext|>
        
        if len(q_tokens) > max_question_len or len(a_tokens) > max_answer_len:
            skipped += 1
            continue
            
        # Pad sequences
        q_tokens.extend([50258] * (max_question_len - len(q_tokens)))
        a_tokens.extend([50258] * (max_answer_len - len(a_tokens)))
        
        if image_id not in image_qa_pairs:
            # Assuming images are in 'images' folder inside questions_dir
            image_qa_pairs[image_id] = {
                'questions': [],
                'answers': [],
                'image_path': os.path.join('images', f"{image_id}.jpg")
            }
        
        image_qa_pairs[image_id]['questions'].append(q_tokens)
        image_qa_pairs[image_id]['answers'].append(a_tokens)
    return image_qa_pairs, skipped


def analyze_sequence_lengths(tokenizer, questions_dir):
    question_lengths = []
    answer_lengths = []
    
    # Process all training question files
    for i in range(2):  # 0-2 for train_all_questions
        file_path = os.path.join(questions_dir, f'train_all_questions_{i}.json')
        
        print(f"\nProcessing {file_path}")
        maxlen = (0, 0)
        with open(file_path, 'r') as f:
            questions_data = json.load(f)
            
        for question_id, question_info in tqdm(questions_data.items()):
            question = question_info['question']
            answer = question_info['fullAnswer']
            
            question_tokens = [50259] + tokenizer.encode(question) + [50260]
            answer_tokens = [50257] + tokenizer.encode(answer) + [50256]
            
            question_lengths.append(len(question_tokens))
            answer_lengths.append(len(answer_tokens))

        maxlen = max(maxlen, (max(question_lengths), max(answer_lengths)))
        print(f"After processing {file_path}, max lengths are: Questions: {maxlen[0]}, Answers: {maxlen[1]}")
            
    
    question_lengths = np.array(question_lengths)
    answer_lengths = np.array(answer_lengths)
    
    print("\nQuestion Length Statistics:")
    print(f"Min length: {question_lengths.min()}")
    print(f"Max length: {question_lengths.max()}")
    print(f"Mean length: {question_lengths.mean():.2f}")
    print(f"95th percentile: {np.percentile(question_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(question_lengths, 99):.2f}")
    
    print("\nAnswer Length Statistics:")
    print(f"Min length: {answer_lengths.min()}")
    print(f"Max length: {answer_lengths.max()}")
    print(f"Mean length: {answer_lengths.mean():.2f}")
    print(f"95th percentile: {np.percentile(answer_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(answer_lengths, 99):.2f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(question_lengths, bins=50, alpha=0.7)
    plt.title('Question Length Distribution')
    plt.xlabel('Length (tokens)')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(answer_lengths, bins=50, alpha=0.7)
    plt.title('Answer Length Distribution')
    plt.xlabel('Length (tokens)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    return question_lengths, answer_lengths

if __name__ == "__main__":
    
    # tokenizer = get_tokenizer()

    # train_annotation_path = "/unity/g1/kgalla/datasets/annotations/captions_train2014.json"
    # train_output_path = "/unity/g1/kgalla/datasets/annotations/coco_train2014_captions.h5"

    # val_annotation_path = "/unity/g1/kgalla/datasets/annotations/captions_val2014.json"
    # val_output_path = "/unity/g1/kgalla/datasets/annotations/coco_val2014_captions.h5"

    # prepare_dataset(tokenizer, train_annotation_path, train_output_path)
    # prepare_dataset(tokenizer, val_annotation_path, val_output_path)

    # train_path = "/unity/g1/kgalla/datasets/vqa_dataset/data_train.csv"
    # eval_path = "/unity/g1/kgalla/datasets/vqa_dataset/data_eval.csv"

    # train_output_path = "/unity/g1/kgalla/datasets/vqa_dataset/vqa_train.h5"
    # eval_output_path = "/unity/g1/kgalla/datasets/vqa_dataset/vqa_eval.h5"

    # print("Analyzing training set:")
    # train_q_lens, train_a_lens = analyze_sequence_lengths(tokenizer, train_path)

    # print("\nAnalyzing evaluation set:")
    # eval_q_lens, eval_a_lens = analyze_sequence_lengths(tokenizer, eval_path)

    # # Get overall maximum length
    # overall_max = max(
    #     train_q_lens.max(), train_a_lens.max(),
    #     eval_q_lens.max(), eval_a_lens.max()
    # )
    # print(f"\nRecommended max_len (maximum length across all sets): {overall_max}")

    tokenizer = get_vqa_tokenizer()
    # prepare_vqa_dataset(tokenizer, train_path, train_output_path)
    # prepare_vqa_dataset(tokenizer, eval_path, eval_output_path)
    # analyze_sequence_lengths(tokenizer, "/unity/g1/kgalla/datasets/gqa//train_all_questions/")

    prepare_gqa_dataset(
        tokenizer=tokenizer,
        questions_dir='/unity/g1/kgalla/datasets/gqa/train_all_questions',
        split='train',
        output_path='/unity/g1/kgalla/datasets/gqa/gqa_train.h5'
    )

    prepare_gqa_dataset(
        tokenizer=tokenizer,
        questions_dir='/unity/g1/kgalla/datasets/gqa',
        split='val', 
        output_path='/unity/g1/kgalla/datasets/gqa/gqa_val.h5'
    )
