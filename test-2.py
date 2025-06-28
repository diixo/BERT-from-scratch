
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

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders
from transformers import BertTokenizerFast



words = []
with open("vocab.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]


# 1️⃣ Создаёшь Tokenizer (НЕ BertTokenizerFast!)
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 2️⃣ Настраиваешь нормализацию и pre-tokenizer
tokenizer.normalizer = normalizers.BertNormalizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
tokenizer.decoder = decoders.WordPiece(prefix="##")

# 3️⃣ Тренер
trainer = trainers.WordPieceTrainer(
    vocab_size=50_000,
    min_frequency=1,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words,
    initial_alphabet=list("abcdefghijklmnopqrstuvwxyz0123456789")
)

# 4️⃣ Учишь на файлах
tokenizer.train(["stub.txt"], trainer)



