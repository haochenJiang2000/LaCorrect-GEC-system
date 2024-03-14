import sys
from model.postprocess.segment import tokenization
from tqdm import tqdm
import argparse
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(dir_path, "vocab.txt"), do_lower_case=False)


def main(args):
    with open(args.data_file, "r", encoding="utf-8") as f1:
        with open(args.char_file, "w", encoding="utf-8") as f2:
            for line in tqdm(f1.read().split("\n")):
                line = line.strip()
                line = line.replace(" ", "")
                line = tokenization.convert_to_unicode(line)
                if not line:
                    continue
                tokens = tokenizer.tokenize(line)
                f2.write(' '.join(tokens)+"\n")


def segment(line):
    line = line.strip()
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        return ''
    tokens = tokenizer.tokenize(line)
    return ' '.join(tokens)


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',
                        help='Path to the data',
                        required=True)
    parser.add_argument('--char_file',
                        help='Path to the char data',
                        required=True)
    args = parser.parse_args()
    main(args)