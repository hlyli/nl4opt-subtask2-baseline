from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig, \
    T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, T5Tokenizer, GenerationConfig, \
    EarlyStoppingCallback, BartTokenizer, BartForConditionalGeneration
from transformers.trainer_utils import set_seed
import parsing_utils.constants as const
import parsers
import jsonlines
import itertools
import torch
import nltk

MAX_TOKEN_LENGTH = 512


class RDFDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def create_dataset(tokenizer, inputs, targets):
    data = tokenizer([x.lower() for x in inputs], max_length=MAX_TOKEN_LENGTH, truncation=True)
    labels = tokenizer([" ".join(x) for x in targets],
                       max_length=MAX_TOKEN_LENGTH, truncation=True)['input_ids']

    # change pad token id to -100 for loss
    # this is only necessary if we pad to max length and do not use DataCollator
    for l in labels:
        for i, tk in enumerate(l):
            if tk == tokenizer.pad_token_id:
                l[i] = -100

    data_as_list = [{
        'input_ids': i,
        'attention_mask': a,
        'labels': l
    }
        for i, a, l in zip(data['input_ids'], data['attention_mask'], labels)]

    return RDFDataset(data_as_list)


def get_jsons(fname, read_document=False):
    with jsonlines.open(fname) as reader:
        examples = [line for line in reader.iter()]
        if read_document:
            parser = parsers.JSONFormulationParser()
            examples = [parser.get_data(ex) for ex in examples]
    return examples


def parse_json(fname):
    parser = parsers.JSONFormulationParser()
    with jsonlines.open(fname) as reader:
        examples = [line for line in reader.iter()]
        parsed = [parser.parse(ex) for ex in examples]

    return parsed


def get_texts(fname):
    parser = parsers.JSONFormulationParser()
    examples = get_jsons(fname)
    texts = [parser.get_text(ex) for ex in examples]

    return texts


def get_all_words(data):
    return list(itertools.chain(*parsers.preprocess_list_of_text(data)))


if __name__ == '__main__':
    # download NLTK packages
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    set_seed(1)

    # pretrained models to fine tune
    # BART works well but outputs strange tokens sometimes
    # T5+ bimodal and regular version work best
    pretrained_name = "Salesforce/codet5p-220m"
    out_dir = "checkpoints_no_preprocess_t5p_base"

    # dataset preprocessing
    train_set = parse_json('../data/train.jsonl')
    dev_set = parse_json('../data/dev.jsonl')

    train_set_rdf = [parsers.convert_to_rdf(ex) for ex in train_set]
    dev_set_rdf = [parsers.convert_to_rdf(ex) for ex in dev_set]

    train_texts = get_texts('../data/train.jsonl')
    dev_texts = get_texts('../data/dev.jsonl')
    test_texts = get_texts('../data/test.jsonl')

    # import model, tokenizer and update tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(pretrained_name, trust_remote_code=True)
    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_name,
        device_map="auto",
        trust_remote_code=True,
        config=config
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    # only add the constants, do not update vocabulary with other words from LPWP dataset
    new_tokens = (
            list(const.TYPE_DICT.values()) +
            [const.LT, const.GT] +
            const.RDF_CONSTANTS
    )

    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
    # need to sort for non-determinism
    new_tokens = list(new_tokens)
    new_tokens.sort()

    # add new tokens to model and resize
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=20,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=1e-2,
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="inverse_sqrt",
        warmup_ratio=0.05,
        seed=1
    )

    train_dataset = create_dataset(tokenizer, train_texts, train_set_rdf)
    dev_dataset = create_dataset(tokenizer, dev_texts, dev_set_rdf)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator
    )

    trainer.train()

    tokenizer.save_pretrained(f'{out_dir}/tokenizer')
