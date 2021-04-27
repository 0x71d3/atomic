import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu

from finetune import read_atomic_split, AtomicDataset


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    test_texts, test_labels = read_atomic_split(f'{args.data_dir}/v4_atomic_tst.csv')
    
    tokenizer = BartTokenizer.from_pretrained(f'./{args.output_dir}')
    
    test_encodings = tokenizer.prepare_seq2seq_batch(test_texts, test_labels, truncation=True, padding=True)

    test_dataset = AtomicDataset(test_encodings)
    
    model = BartForConditionalGeneration.from_pretrained(f'./{args.output_dir}')
    model.to(device)
    model.eval()

    list_of_references = []
    hypotheses = []

    test_loader = DataLoader(test_dataset, batch_size=64)
    for batch in tqdm(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        beam_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )

        for beam_output in beam_outputs:
            list_of_references.append([word_tokenize(tokenizer.decode(beam_output, skip_special_tokens=True))])

        labels = batch['labels'].to(device)
        for label in labels:
            hypotheses.append(word_tokenize(tokenizer.decode(label, skip_special_tokens=True)))

    print(corpus_bleu(list_of_references, hypotheses, weights=(0.5, 0.5, 0, 0)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='atomic_data')
    parser.add_argument('--output_dir', default='comet_large')

    args = parser.parse_args()

    main(args)
