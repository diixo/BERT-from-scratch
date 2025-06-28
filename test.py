
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.pre_tokenizers import ByteLevel, BertPreTokenizer
from transformers import BertTokenizerFast
from tokenizers.normalizers import BertNormalizer
from transformers import BertTokenizer
from tokenizers.implementations import BertWordPieceTokenizer



tokenizer_json = "bert_tokenizer-00.json"


def train_tokenizer():

    words = []
    with open("vocab.txt", "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # 2. Train WordPiece tokenizer
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.normalizer = BertNormalizer(clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True,
    )

    #initial_alphabet = ByteLevel.alphabet()
    initial_alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789")
    ext_alphabet = ["##a", "##b", "##c", "##d", "##e", "##f", "##g", "##h", "##i", "##j", "##k", "##l", "##m",
                    "##n", "##o", "##p", "##q", "##r", "##s", "##t", "##u", "##v", "##w", "##x", "##y", "##z"]

    trainer = trainers.WordPieceTrainer(
        vocab_size=100_000,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + ext_alphabet,
        initial_alphabet=initial_alphabet,
    )

    tokenizer.train(files=["stub.txt"], trainer=trainer)

    tokenizer.save(tokenizer_json)
    print(f"Tokenizer saved to {tokenizer_json}")
    return tokenizer


train_tokenizer()


fast_tokenizer = BertTokenizerFast(tokenizer_file=tokenizer_json)

fast_tokenizer.save_pretrained("./bert-small-tokenizer")


test_txt = "The evolution of a process is directed by a pattern of rules called a [MASK]"

print("Tokens:", fast_tokenizer.tokenize(test_txt))
print("Tokens:", fast_tokenizer.tokenize("The"))

