import parsers
import scoring
import jsonlines
import numpy as np
import parsing_utils.constants as const

from argparse import ArgumentParser

DIVIDER = '|'
NEWLINE = '<NEWLINE>'
VARIABLE = 'var'
LHS = 'lhs'
RHS = 'rhs'


def to_string(problem, as_text):
    rows = []
    col_lhs = ''
    col_rhs = ''
    value_to_entity = {}
    # reverse dictionary
    for k,v in problem.entities.items():
        if v not in value_to_entity:
            value_to_entity[v] = k

    for k,v in value_to_entity.items():
        col_lhs = " ".join([col_lhs, f'{v}', LHS, DIVIDER])
        col_rhs = " ".join([col_rhs, f'{v}', RHS, DIVIDER])

    col_lhs = " ".join([col_lhs, 'const', LHS])
    col_rhs = " ".join([col_rhs, 'const', RHS])
    rows.append(" ".join([DIVIDER, DIVIDER, col_lhs, DIVIDER, col_rhs]))
    if as_text:
        objective, constraints = parsers.convert_to_canonical_two_sided_wordform(problem)
    else:
        objective, constraints = parsers.convert_to_canonical_two_sided(problem)

    rows.append(" ".join(['objective', DIVIDER, f" {DIVIDER} ".join(map(str, objective))]))
    for i, constraint in enumerate(constraints):
        row_header = f'constraint_{i}'
        entries = f" {DIVIDER} ".join(map(str, constraint))
        rows.append(" ".join([row_header, DIVIDER, entries]))

    return f" {DIVIDER} {NEWLINE} {DIVIDER} ".join(rows) + f' {DIVIDER} '


if __name__ == '__main__':
    cli = ArgumentParser()
    cli.add_argument('-f', '--file', type=str, default="")
    cli.add_argument('-t', '--text', action='store_true')

    args = cli.parse_args()

    with jsonlines.open(args.file) as reader:
        parser = parsers.JSONFormulationParser()
        examples = [line for line in reader.iter()]
        parsed = [parser.parse(ex) for ex in examples]
        # parsed = [ex for ex in parsed if ex is not None]

        problem_texts = []

        for example in examples:
            for k in example.keys():
                problem_texts.append(example[k]['document'])

        stringified = []
        for i in range(len(parsed)):
            stringified.append(to_string(parsed[i], args.text))

        out_fname = args.file.split('/')[-1].split('.')[0]
        out_dir = "/".join(args.file.split('/')[:-1])

        if args.text:
            out_dir += '/text'
        else:
            out_dir += '/numerical'
        with open(f'{out_dir}/{out_fname}.text', 'w', encoding='utf8') as out:
            out.write("\n".join(problem_texts))
        with open(f'{out_dir}/{out_fname}.data', 'w') as out:
            out.write("\n".join(stringified))