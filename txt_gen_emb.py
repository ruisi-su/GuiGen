import os
import random
import argparse
import torch
import numpy as np
import Core.Constants as Constants
from bpemb import BPEmb
import fasttext
import fasttext.util
from tqdm import tqdm


def parse_args():
    """
    Wrapper function of argument parsing process.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--emb_save_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_emb.pt'),
        help='embedding data save location.'
    )
    parser.add_argument(
        '--dict_load_dir', type=str, default=os.path.join(Constants.TRAIN_PATH, 'para_train_dict.pt'),
        help='token to index dictionary load location.'
    )
    parser.add_argument(
        '--dim', type=str, default=100, help='embedding size'
    )

    args = parser.parse_args()
    return args


def gen_word_vec(dictionary, bpemb, ftemb, output):
    w2v = dict()
    bpemb_exist = bpe_nonexist = 0
    for word in tqdm(dictionary):
        try:
            word_emb = bpemb[word]
            bpemb_exist += 1
        except KeyError as e:
            # print(e)
            word_decode = ''.join(word).replace('▁', ' ')
            word_emb = ftemb.get_word_vector(word_decode)
            bpe_nonexist += 1
        w2v[word] = word_emb
    print(f'[Warning] {bpe_nonexist} words do not exist in bpembed out of {bpe_nonexist + bpemb_exist} words')
    torch.save(w2v, output)


def main():
    """ Main function """

    args = parse_args()

    if os.path.exists(args.dict_load_dir):
        print('[Info] Loading dictionary file')
        w2i = torch.load(args.dict_load_dir)
        txt_word2idx = w2i['text']
    else:
        raise ValueError('[Info] Invalid dictionary location.')

    bpemb_en = BPEmb(lang="en", dim=args.dim)
    ft = fasttext.load_model('../fasttext/cc.en.300.bin')
    fasttext.util.reduce_model(ft, args.dim)
    gen_word_vec(txt_word2idx, bpemb_en, ft, args.emb_save_dir)

    # torch.save(w2i_dict, args.dict_save_dir)
    print('[Info] Finished.')


if __name__ == '__main__':
    main()
