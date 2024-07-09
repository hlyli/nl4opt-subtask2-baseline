import parsers
import scoring
import jsonlines
import numpy as np
import parsing_utils.constants as const
import word2number.w2n as w2n

from argparse import ArgumentParser

DIVIDER = '|'
NEWLINE = '<NEWLINE>'
VARIABLE = 'var'
LHS = 'lhs'
RHS = 'rhs'


def to_canonical(text):
    try:
        parser = parsers.Parser()
        rows = text.split(NEWLINE)
        rows = [row.split(DIVIDER) for row in rows]

        matrix = rows[1:]
        # strip out space, constraint, and objective tags
        matrix = [row[2:-1] for row in matrix]

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] = parser.parse_number(matrix[i][j].strip())

        objective = np.array(matrix[0])
        constraints = np.array(matrix[1:])
        width = objective.shape[0]

        objective_lhs = objective[:width//2]
        objective_rhs = objective[width//2:]

        objective_out = objective_lhs - objective_rhs

        constraints_lhs = constraints[:, :width//2]
        constraints_rhs = constraints[:, width//2:]

        constraints_out = constraints_lhs - constraints_rhs

        constraints_out[:, -1] = -constraints_out[:, -1]
    except IndexError as e:
        print(f"Error: {e} in parsing")
        return np.array([]), np.array([])
    except ValueError as e:
        print(f"Error: {e} in parsing")
        return np.array([]), np.array([])

    return objective_out[:-1], constraints_out, rows[0]


def to_string(problem, as_text):
    rows = []
    col_lhs = ''
    col_rhs = ''
    value_to_entity = {}
    # reverse dictionary
    for k, v in problem.entities.items():
        if v not in value_to_entity:
            value_to_entity[v] = k

    for k, v in value_to_entity.items():
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
    cli.add_argument('-p', '--predictions', type=str, default="")

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

    predictions = []

    with open(args.predictions, 'r') as reader:
        line = reader.readline()
        while line is not None and line != '':
            predictions.append(to_canonical(line))
            line = reader.readline()

    gt_objectives = []
    gt_constraints = []
    pred_objectives = []
    pred_constraints = []

    for gt, pred in zip(parsed, predictions):
        canonical = parsers.convert_to_canonical(gt)
        gt_objectives.append(canonical.objective)
        gt_constraints.append(canonical.constraints)
        pred_objectives.append(pred[0])
        pred_constraints.append(pred[1])
        # print(gt.entities, pred[2])

    print(scoring.overall_score(pred_objectives, pred_constraints, gt_objectives, gt_constraints))
