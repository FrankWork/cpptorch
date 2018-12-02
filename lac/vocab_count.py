import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
        '--input', type=str,  default=None, help='dir or file')
parser.add_argument(
        '--output', type=str,  default=None, help='dir or file')
args = parser.parse_args()

vocab_freq = {}
with open(args.input, 'r') as f:
    for line in tqdm(f):
        for tok in line.strip().split():
            if tok not in vocab_freq:
                vocab_freq[tok] = 1
            else:
                vocab_freq[tok] += 1

vocab_freq = [(tok, freq) for (tok, freq) in vocab_freq.items()]
vocab_freq = sorted(vocab_freq, key=lambda x:x[1], reverse=True)

with open(args.output, 'w') as f:
    for (tok, freq) in tqdm(vocab_freq):
        f.write('{}\t{}\n'.format(tok, freq))

print(len(vocab_freq))

        

