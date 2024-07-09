from transformers import AutoTokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
from train_rdf import *
import nltk
from pathlib import Path


def eval_dataset(model, tokenizer, dataset, out_name):
    # experimental stuff for bad words to block out random tokens in BART generation
    # bad_words = '"\'()’'
    # bad_words_ids = [tokenizer.encode(x, add_special_tokens=False) for x in bad_words]

    preprocessed = parsers.preprocess_list_of_text(dataset, join=True)
    outputs = []
    inputs = []
    for ex in tqdm(preprocessed):
        model_input = tokenizer.encode(ex, return_tensors='pt').to('cuda')

        model_output = model.generate(tokenizer.encode(ex.lower(), return_tensors='pt').to('cuda'),
                                      max_new_tokens=512,
                                      num_beams=5,
                                      num_return_sequences=1,
                                      repetition_penalty=1.0,
                                      #   no_repeat_ngram_size=4,
                                      #   bos_token_id=tokenizer.bos_token_id,
                                      #   bad_words_ids=bad_words_ids,
                                      early_stopping=True)

        inputs.append(tokenizer.decode(model_input[0], skip_special_tokens=True))
        outputs.append(tokenizer.decode(model_output[0], skip_special_tokens=True))

    with open(out_name, 'w') as f:
        f.write("\n".join(outputs))

    with open(f'{out_name}.in', 'w') as f:
        f.write("\n".join(inputs))


if __name__ == '__main__':
    suffix = 'no_preprocess_t5p_base'
    out_folder = f'eval_out_{suffix}'
    checkpoint_folder = f'checkpoints_{suffix}'
    checkpoint_number = 360

    Path(out_folder).mkdir(exist_ok=True)

    train_texts = get_texts('../data/train.jsonl')
    dev_texts = get_texts('../data/dev.jsonl')
    test_texts = get_texts('../data/test.jsonl')

    tokenizer = AutoTokenizer.from_pretrained(f'{checkpoint_folder}/tokenizer')

    model = T5ForConditionalGeneration.from_pretrained(
        f'{checkpoint_folder}/checkpoint-{checkpoint_number}',
        device_map="auto"
    )

    eval_dataset(model, tokenizer, test_texts, f'{out_folder}/test.rdf')
    # eval_dataset(model, tokenizer, train_texts, f'{out_folder}/train.rdf')
    # eval_dataset(model, tokenizer, dev_texts, f'{out_folder}/dev.rdf')
