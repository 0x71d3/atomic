import argparse
import json

import pandas as pd
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

tokens = [
    'oEffect',
    'oReact',
    'oWant',
    'xAttr',
    'xEffect',
    'xIntent',
    'xNeed',
    'xReact',
    'xWant',
]


def read_atomic_split(split_path):
    texts = []
    labels = []

    df = pd.read_csv(split_path, index_col=0)
    df.iloc[:,:9] = df.iloc[:,:9].apply(lambda col: col.apply(json.loads))

    for index, series in df.iterrows():
        for column in df.columns[:9]:
            if series[column]:
                for label in series[column]:                    
                    texts.append(f'{index} {column}')
                    labels.append(label)
    
    return texts, labels


class AtomicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def main(args):
    train_texts, train_labels = read_atomic_split(f'{args.data_dir}/v4_atomic_trn.csv')
    val_texts, val_labels = read_atomic_split(f'{args.data_dir}/v4_atomic_dev.csv')
    # test_texts, test_labels = read_atomic_split(f'{args.data_dir}/v4_atomic_tst.csv')

    tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    num_added_toks = tokenizer.add_tokens(tokens)

    train_encodings = tokenizer.prepare_seq2seq_batch(train_texts, train_labels, truncation=True, padding=True)
    val_encodings = tokenizer.prepare_seq2seq_batch(val_texts, val_labels, truncation=True, padding=True)
    # test_encodings = tokenizer.prepare_seq2seq_batch(test_texts, test_labels, truncation=True, padding=True)

    train_dataset = AtomicDataset(train_encodings)
    val_dataset = AtomicDataset(val_encodings)
    # test_dataset = AtomicDataset(test_encodings)

    model = BartForConditionalGeneration.from_pretrained(args.pretrained_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f'./results/{args.output_dir}',
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        max_grad_norm=0.1,
        num_train_epochs=1,
        warmup_steps=500,
        logging_dir=f'./logs/{args.output_dir}',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='atomic_data')
    parser.add_argument('--output_dir', default='comet_large')

    parser.add_argument('--pretrained_model_name_or_path', default='facebook/bart-large')
    parser.add_argument('--per_device_batch_size', default=2, type=int)

    args = parser.parse_args()

    main(args)
