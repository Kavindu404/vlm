import json
import h5py
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd
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

def prepare_vqa_dataset(tokenizer, data_path, output_path, max_question_len=48, max_answer_len=48):
    # Read the text file
    qa_pairs = []
    with open(data_path, 'r') as f:
        next(f)
        for line in f:

            question_answer, image_id = line.rsplit(',', 1)

            question, answer = question_answer.split(',', 1)
            qa_pairs.append({
                'question': question.strip(),
                'answer': answer.strip(),
                'image_id': image_id.strip()
            })
    
    with h5py.File(output_path, 'w') as h5f:
        questions_group = h5f.create_group('questions')
        answers_group = h5f.create_group('answers')
        metadata = h5f.create_group('metadata')

        metadata.attrs['max_question_length'] = max_question_len
        metadata.attrs['max_answer_length'] = max_answer_len
        metadata.attrs['pad_token'] = 50258  
        metadata.attrs['start_token'] = 50257  
        metadata.attrs['eos_token'] = 50256 
        metadata.attrs['question_start'] = 50259  
        metadata.attrs['question_end'] = 50260 
        
        image_qa_pairs = {}
        skipped = 0
        
        for item in tqdm(qa_pairs):
            image_id = item['image_id']
            question = item['question']
            answer = item['answer']
            
            # Encode question: <startofquestion> + question + <endofquestion> + <pad>
            q_tokens = [50259]  # <|startofquestion|>
            q_tokens += tokenizer.encode(question)
            q_tokens += [50260]  # <|endofquestion|>
            
            # Encode answer: <startoftext> + answer + <endoftext> + <pad>
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
                image_qa_pairs[image_id] = {
                    'questions': [],
                    'answers': []
                }
            image_qa_pairs[image_id]['questions'].append(q_tokens)
            image_qa_pairs[image_id]['answers'].append(a_tokens)
        
        print(f"Skipped {skipped} sequences longer than max length")

        for image_id, qa_data in tqdm(image_qa_pairs.items(), desc="Saving to H5"):
            questions_array = np.array(qa_data['questions'], dtype=np.int32)
            answers_array = np.array(qa_data['answers'], dtype=np.int32)
            
            questions_group.create_dataset(
                str(image_id),
                data=questions_array,
                compression='gzip',
                compression_opts=9
            )
            
            answers_group.create_dataset(
                str(image_id),
                data=answers_array,
                compression='gzip',
                compression_opts=9
            )
        
        metadata.create_dataset(
            'image_ids',
            data=np.array(list(image_qa_pairs.keys()), dtype='S')
        )
        
        metadata.attrs['num_image_tokens'] = 196  
        metadata.attrs['total_sequence_length'] = 196 + max_question_len 


def analyze_sequence_lengths(tokenizer, csv_path):

    data = pd.read_csv(csv_path)
    
    question_lengths = []
    answer_lengths = []
    
    for _, row in tqdm(data.iterrows(), total=len(data)):
        question = row['question']
        answer = row['answer']
        
        question_tokens = [50259] + tokenizer.encode(question) + [50260]
        answer_tokens = [50257] + tokenizer.encode(answer) + [50256]
        
        question_lengths.append(len(question_tokens))
        answer_lengths.append(len(answer_tokens))
    
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

    train_path = "/unity/g1/kgalla/datasets/vqa_dataset/data_train.csv"
    eval_path = "/unity/g1/kgalla/datasets/vqa_dataset/data_eval.csv"

    train_output_path = "/unity/g1/kgalla/datasets/vqa_dataset/vqa_train.h5"
    eval_output_path = "/unity/g1/kgalla/datasets/vqa_dataset/vqa_eval.h5"

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
    prepare_vqa_dataset(tokenizer, train_path, train_output_path)
    prepare_vqa_dataset(tokenizer, eval_path, eval_output_path)

