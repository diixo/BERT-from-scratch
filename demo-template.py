
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import ByteLevel, BertPreTokenizer
from transformers import BertTokenizerFast
from tokenizers.normalizers import BertNormalizer


outpath = "tmp/output-3.txt"


def create_train_corpus():
    vocab = "data/tokens.txt"

    words = []
    with open(vocab, "r", encoding="utf-8") as f:
        words = sorted([line.strip() for line in f if line.strip()])

    filepath = "tmp/corpus-triple-words.txt"
    count = 0
    with open(filepath, "w", encoding="utf-8") as f_out:
        for w in words:
            if w.find("-") < 0:
                f_out.write(f"{w} {w} {w}\n")
                count += 1
    return filepath



def vocab_tokenizer():

    # 2. Train WordPiece tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.normalizer = BertNormalizer(
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True,
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")


    initial_alphabet = ByteLevel.alphabet()
    initial_alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789-+_")
    ext_alphabet = ["##a", "##b", "##c", "##d", "##e", "##f", "##g", "##h", "##i", "##j", "##k", "##l", "##m",
                    "##n", "##o", "##p", "##q", "##r", "##s", "##t", "##u", "##v", "##w", "##x", "##y", "##z"]

    trainer = trainers.WordPieceTrainer(
        vocab_size=50_000,
        min_frequency=3,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + ext_alphabet,
        initial_alphabet=initial_alphabet,
    )

    tokenizer.train(files=[], trainer=trainer)
    #tokenizer.add_tokens(words)

    tokenizer_file = "wordpiece-tokenizer-template.json"
    tokenizer.save(tokenizer_file)
    print(f"Tokenizer saved to {tokenizer_file}")
    return tokenizer_file


# 3. Load tokenizer into Hugging-Face format
hf_tokenizer = BertTokenizerFast(
    tokenizer_file=vocab_tokenizer(),
    pad_token="[PAD]",
    unk_token="[UNK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# hf_tokenizer.save_pretrained("./bert_small_tokenizer")


def tokens_to_file():
    with open("data/db-full.txt", "r", encoding="utf-8") as f:
        word_set = set([line.strip() for line in f if line.strip()])
    word_set = sorted(word_set)

    with open(outpath, "w", encoding="utf-8") as f_out:
        for w in word_set:
            if w.find("-") < 0:
                f_out.write(f"{w}: {str(hf_tokenizer.tokenize(w))}\n")


#tokens_to_file()

