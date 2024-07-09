import parsers
import scoring
from typing import Optional, Dict, List, Tuple
from train_rdf import parse_json
import jsonlines
from collections import defaultdict

def compute_score(pred: List[parsers.ProblemFormulation],
                  label: List[parsers.ProblemFormulation]) -> float:
    pred_canonicals = []
    label_canonicals = []

    for p, l in zip(pred, label):
        pred_canonicals.append(parsers.convert_to_canonical(p))
        label_canonicals.append(parsers.convert_to_canonical(l))

    return scoring.overall_score(
        [x.objective for x in pred_canonicals],
        [x.constraints for x in pred_canonicals],
        [x.objective for x in label_canonicals],
        [x.constraints for x in label_canonicals],
    )


def parse_rdf(fname, order_mappings):
    parser = parsers.RDFParser()
    with open(fname, 'r') as reader:
        examples = [line for line in reader]
        parsed = [parser.parse(ex, mapping) for ex, mapping in zip(examples, order_mappings)]


    return parsed


if __name__ == '__main__':
    labels = parse_json('../data/test.jsonl')
    parser = parsers.RDFParser()
    # predictions = [parser.parse(" ".join(parsers.convert_to_rdf(ex)), mapping) for ex, mapping in zip(labels, [x.entities for x in labels])]
    predictions = parse_rdf('eval_out_no_preprocess_t5p_base/test.rdf', [x.entities for x in labels])

    num_obj = len(labels)
    num_const = sum([len(l.constraints) for l in labels])
    num_non_triple_obj = sum([p.num_exc_of_type(parsers.ObjectiveNotTriple) for p in predictions])
    num_non_triple_const = sum([p.num_exc_of_type(parsers.ConstraintNotTriple) for p in predictions])

    wrong_const_type_sum = 0

    for p, l in zip(predictions, labels):
        dict_const_type_p = defaultdict(int)
        dict_const_type_l = defaultdict(int)

        for c in p.constraints:
            dict_const_type_p[c.type] += 1
        for c in l.constraints:
            dict_const_type_l[c.type] += 1

        for k in dict_const_type_p:
            if k in dict_const_type_l:
                dict_const_type_l[k] -= 1

        wrong_const_type_sum += sum(dict_const_type_l.values())

    print(f"Num non triple obj {num_non_triple_obj/num_obj}")
    print(f"Num non triple const {num_non_triple_const/num_const}")
    print(f"Num wrong const type {wrong_const_type_sum/num_const}")

    print(compute_score(predictions, labels))
