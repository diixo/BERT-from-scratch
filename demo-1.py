
import torch
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast, BertConfig, BertForMaskedLM, BertTokenizerFast
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
from tokenizers.pre_tokenizers import ByteLevel, BertPreTokenizer
from tokenizers.normalizers import BertNormalizer

import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# 1. Create simple corpus with 5 sentences
sentences = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules called a program.
People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells."""

sentences = sentences.split('\n')
corpus_path = "corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for line in sentences:
        f.write(line + "\n")

corpus = 5 * sentences


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
    initial_alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789")
    ext_alphabet = ["##a", "##b", "##c", "##d", "##e", "##f", "##g", "##h", "##i", "##j", "##k", "##l", "##m",
                    "##n", "##o", "##p", "##q", "##r", "##s", "##t", "##u", "##v", "##w", "##x", "##y", "##z"]

    trainer = trainers.WordPieceTrainer(
        vocab_size=50_000,
        min_frequency=3,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + ext_alphabet,
        initial_alphabet=initial_alphabet,
    )


    tokenizer.train(files=['training.txt'], trainer=trainer)
    #tokenizer.add_tokens(words)

    tokenizer_file = "wordpiece-1-tokenizer.json"
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


# 4. Create config with BERT model
config = BertConfig(
    vocab_size=hf_tokenizer.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=256,
)

model = BertForMaskedLM(config)


# 5. Create simple dataset
class SmallTextDataset(Dataset):
    def __init__(self, texts, tokenizer: PreTrainedTokenizerFast):
        self.examples = [tokenizer.encode(t, max_length=32, truncation=True, padding='max_length') for t in texts]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }

dataset = SmallTextDataset(corpus, hf_tokenizer)


# 6. Create collator for MLM (masking random tokens)
data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tokenizer, mlm=True, mlm_probability=0.15)


# 7. Tune params for learning
training_args = TrainingArguments(
    output_dir="./bert_small",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=4,
    prediction_loss_only=True,
    logging_steps=5,
    learning_rate=1e-4,
)


# 8. Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


# 9. Train
trainer.train()


# (optionally) Save trained model with tokenizer
#trainer.save_model("./bert_small")
print("Training finished and model saved.")


# 10. Test pipeline by prediction the masking word
def test_pipeline(test_txt: str, tokenizer: BertTokenizerFast, model: BertForMaskedLM):

    inputs = tokenizer(test_txt, return_tensors="pt")

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # pass data through a model 
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Logits for [MASK]
    mask_token_logits = logits[0, mask_token_index, :]

    # Top-5 tokens
    top_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

    for sent in sentences:
        print("Tokens:", hf_tokenizer.tokenize(sent))
    print("########################################")

    # Decoding
    print("Top predictions:")
    for token in top_tokens:
        print(f"{tokenizer.decode([token])}")


test_txt = "The evolution of a process is directed by a pattern of rules called a [MASK]"
test_pipeline(test_txt, hf_tokenizer, model.cpu())
