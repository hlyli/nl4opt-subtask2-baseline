import difflib
import xml.etree.ElementTree as ET
import numpy as np
from dataclasses import dataclass, field
from collections import OrderedDict
import re
import parsing_utils.constants as const
import word2number.w2n as w2n
from typing import Optional, Dict, List
import logging
from bs4 import BeautifulSoup
import lxml
import pdb
import itertools
import nltk
import string


# dataclass that stores each term in a declaration


@dataclass
class Term:
    # original name
    name: str
    # index in order mapping
    index: int
    # the constant that multiplies the term
    value: float = None
    value_str: str = ''


@dataclass
class Declaration:
    # minimize/maximize or <=/>=
    direction: str
    # mapping of term name to each term involved in the declaration
    terms: Dict[str, Term]
    # order mapping
    entities: Dict[str, int]


@dataclass
class ObjectiveDeclaration(Declaration):
    # name of variable being minimized/maximized
    name: str


@dataclass
class ConstraintDeclaration(Declaration):
    # type of constraint, e.g., linear
    type: str
    # the constant on the right side of the constraint
    limit: float
    # >= or <= for constraint
    operator: str
    # using ordered dict here as order determines order of operations in balance control constraints
    terms: Dict[str, Term]
    limit_str: str = ''


@dataclass
class ProblemFormulation:
    objective: ObjectiveDeclaration
    constraints: List[ConstraintDeclaration]
    # order mapping mapping each entity to its index
    entities: Dict[str, int]
    exceptions: List[Exception] = field(default_factory=list)

    def num_exc_of_type(self, exc_type):
        total = 0
        for exc in self.exceptions:
            if isinstance(exc, exc_type):
                total += 1
        return total


@dataclass
class CanonicalFormulation:
    objective: np.ndarray
    constraints: np.ndarray


def preprocess_text(input_text, join=False):
    wnl = nltk.stem.WordNetLemmatizer()

    input_text = input_text.lower()
    # input_text = input_text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(input_text)
    # words = [wnl.lemmatize(w) for w in words]

    if join:
        return " ".join(words)
    else:
        return words


def preprocess_list_of_text(input_texts, join=False):
    return [preprocess_text(x, join) for x in input_texts]


def convert_to_canonical(formulation: ProblemFormulation) -> CanonicalFormulation:
    n_entities = max(formulation.entities.values()) + 1 if len(formulation.entities) else 0
    constraints = []
    objective = np.zeros(n_entities)
    for k, v in formulation.objective.terms.items():
        # check if value was given in formulation
        objective[v.index] = v.value if v.value is not None else np.nan

    for constraint in formulation.constraints:

        row = np.ones(n_entities + 1)
        # compute everything as <= at first
        if constraint.type == const.SUM_CONSTRAINT:
            # x + y <= 150
            row[-1] = constraint.limit

        elif constraint.type == const.LOWER_BOUND or constraint.type == const.UPPER_BOUND:
            # x <= 50
            # compute as upper bound and flip later for lower bound if necessary
            row *= 0
            for k, v in constraint.terms.items():
                row[v.index] = 1
            row[-1] = constraint.limit
        elif constraint.type == const.LINEAR_CONSTRAINT:
            # 2x + 3y <= 20
            for k, v in constraint.terms.items():
                # check if value was given in formulation
                row[v.index] = v.value if v.value is not None else np.nan
            row[-1] = constraint.limit
        elif constraint.type == const.RATIO_CONTROL_CONSTRAINT:
            # x <= 0.7 (x + y)
            row *= -constraint.limit
            for k, v in constraint.terms.items():
                row[v.index] = 1 - constraint.limit
            row[-1] = 0
        elif constraint.type == const.BALANCE_CONSTRAINT_1:
            # x <= 3y
            row *= 0
            # y should be first term, but we will parse whichever term that has value as y
            for i, (k, v) in enumerate(constraint.terms.items()):
                if v.value is not None:
                    row[v.index] = - v.value
                else:
                    row[v.index] = 1
        elif constraint.type == const.BALANCE_CONSTRAINT_2:
            # x <= y
            for i, (k, v) in enumerate(constraint.terms.items()):
                # y is first term
                if i == 0:
                    row[v.index] = - 1
                elif i == 1:
                    row[v.index] = 1
        # flip if >=
        if constraint.operator == const.GT:
            row *= -1
        constraints.append(row)

    return CanonicalFormulation(objective, np.asarray(constraints))


def convert_to_rdf(formulation: ProblemFormulation) -> list[str]:
    statements = []

    def append_statement(subject, predicate, obj):
        assert isinstance(subject, str)
        assert isinstance(predicate, str) or isinstance(predicate, (int, float, complex))
        assert isinstance(obj, str) or isinstance(obj, (int, float, complex))

        subject = preprocess_text(subject)
        predicate = str(predicate)
        predicate = preprocess_text(predicate) if predicate not in const.RDF_CONSTANTS else [predicate]
        obj = str(obj)
        obj = preprocess_text(obj) if obj not in const.RDF_CONSTANTS else [obj]

        statements.append([*subject,
                           const.RDF_DELIMITER,
                           *predicate,
                           const.RDF_DELIMITER,
                           *obj,
                           const.RDF_LINE_SEP])

    def finish_declaration():
        statements.append([const.RDF_DECLARATION_SEP])

    # objective
    append_statement(formulation.objective.name, const.RDF_OBJ_TYPE, formulation.objective.direction)
    for term in formulation.objective.terms.values():
        append_statement(formulation.objective.name, term.value_str, term.name)
    finish_declaration()

    # constraints
    for constraint in formulation.constraints:
        append_statement(constraint.direction, const.RDF_CONST_TYPE, constraint.type)
        append_statement(constraint.direction, const.RDF_CONST_LIM, constraint.limit_str)
        append_statement(constraint.direction, const.RDF_OPERATOR, constraint.operator)

        if constraint.type == const.SUM_CONSTRAINT:
            # nothing to be done for sum
            pass
        elif constraint.type == const.LOWER_BOUND or constraint.type == const.UPPER_BOUND:
            for term in constraint.terms.values():
                append_statement(constraint.direction, const.RDF_VAR, term.name)
        elif constraint.type == const.LINEAR_CONSTRAINT:
            for k, v in constraint.terms.items():
                append_statement(constraint.direction, v.value_str, v.name)
        elif constraint.type == const.RATIO_CONTROL_CONSTRAINT:
            for term in constraint.terms.values():
                append_statement(constraint.direction, const.RDF_VAR, term.name)
        elif constraint.type == const.BALANCE_CONSTRAINT_1:
            for term in constraint.terms.values():
                # treat y as the one with an attached value
                if term.value_str:
                    append_statement(constraint.direction, term.value_str, term.name)
                else:
                    append_statement(constraint.direction, const.RDF_XBY_X, term.name)
        elif constraint.type == const.BALANCE_CONSTRAINT_2:
            # x <= y
            for i, term in enumerate(constraint.terms.values()):
                # y is first term
                if i == 0:
                    append_statement(constraint.direction, const.RDF_XBY_Y, term.name)
                elif i == 1:
                    append_statement(constraint.direction, const.RDF_XBY_X, term.name)

        finish_declaration()

    return list(itertools.chain(*statements))


class Parser:
    def __init__(self, print_errors=True) -> None:
        self.print_errors = print_errors

    def parse_number(self, x: str) -> float:
        """

        :param x: any number-like string
        :return: best-effort attempt at converting the string to a number; returns 0 if it could not parse
        """
        x = x.strip().replace(',', '').strip(';,$')
        # remove extra periods from the end
        x = re.sub('\.+$', '', x)
        multiplier = 1
        x_out = 0

        # convert percent, cents to fractional
        x_sub = re.sub('(percent)|%|¢', '', x)
        if x_sub != x:
            multiplier = 1 / 100
            x = x_sub
        try:
            x_out = float(x) * multiplier
        except ValueError:
            # see if there are any digits and strip everything else out
            if re.search('[\d.]+', x):
                res = re.sub('[^0-9.]', '', x)
                x_out = float(res) * multiplier
                if self.print_errors:
                    logging.info(
                        f'Non-numeric input \"{x}\" converted to \"{x_out}\" by filtering out non-number characters')
            else:
                try:
                    # see if it is in predefined constants
                    if x in const.NUMS_DICT:
                        x_out = const.NUMS_DICT[x] * multiplier
                    else:
                        x_out = float(w2n.word_to_num(x)) * multiplier
                        if self.print_errors:
                            logging.info(f'Non-numeric input \"{x}\" converted to {x_out} with w2n')
                except ValueError:
                    if self.print_errors:
                        logging.warning(f'Could not convert word \"{x}\" to number')
        return x_out

    # for general text parsing
    def parse_text(self, x: str) -> str:
        return x.strip()

    # for fuzzy searching the order mapping
    # will only strip if order mapping is empty
    def parse_entity(self, x: str, order_mapping: dict) -> str:
        x = x.strip()
        if x in order_mapping:
            return x
        elif len(order_mapping):
            # try to find closest match in the order mapping
            best_similarity = 0
            best_match = x
            for k in order_mapping:
                # use lower case as this metric penalizes for different cases
                score = self.similarity(x, k)
                if score > best_similarity:
                    best_similarity = score
                    best_match = k
            return best_match
        else:
            return x

    # string similarity metric to perform fuzzy search
    def similarity(self, a: str, b: str):
        a = a.strip().lower()
        b = b.strip().lower()
        # ignore spaces with isjunk
        sm = difflib.SequenceMatcher(isjunk=lambda x: x in " \t", a=a, b=b)
        return sm.ratio()

    # to be overridden if a custom parser is required
    def parse(self, data: object, order_mapping: Optional[dict] = None) -> ProblemFormulation:
        """

        :param data: parsing_utils to parse, typically a string or dict
        :param order_mapping: mapping of variables to an index to convert into canonical form; should be given in dataset
        """
        pass


# Parses the XML-like intermediate outputs of a model
class ModelOutputXMLParser(Parser):
    def xmltree(self, data: str) -> Optional[ET.Element]:
        # fix mismatched tags
        bs = BeautifulSoup(f'<s>{data}</s>', 'xml')
        # remove empty elements
        for x in bs.find_all():
            if len(x.get_text(strip=True)) == 0:
                x.extract()
        return ET.fromstring(str(bs))

    def parse(self, data: str, order_mapping=None) -> ProblemFormulation:
        try:
            root = self.xmltree(data)
            # use iter instead of find_all in case of weird nesting
            declarations = root.iter('DECLARATION')
            # assuming one objective
            objective = None
            constraints = []
            # find objective first
            for declaration in declarations:
                if declaration.find('OBJ_DIR') is not None:
                    objective = self.parse_objective(declaration, order_mapping)
                    break
            # then do the constraints
            for declaration in declarations:
                if declaration.find('CONST_DIR') is not None:
                    try:
                        constraints.append(self.parse_constraint(declaration, objective.entities))
                    except ValueError as e:
                        if self.print_errors:
                            logging.warning(f'Could not parse constraint, skipping: {e}')
            return ProblemFormulation(objective, constraints, entities=objective.entities)
        except Exception as e:
            if self.print_errors:
                logging.warning(
                    f'Could not parse text \"{data}\".\nPlease check that your model output is parsable XML.\nProceeding with empty ProblemFormulation.')
            # cannot be parsed, returning an empty ProblemFormulation
            return ProblemFormulation(ObjectiveDeclaration('', {}, {}, ''), [], entities={})

    def parse_objective(self, root: ET.Element, order_mapping) -> ObjectiveDeclaration:
        obj_dir = ''
        obj_name = ''
        variables = {}
        entities = order_mapping if order_mapping is not None else {}
        current_var = None
        count = 0
        for node in root:
            if node.tag == 'OBJ_DIR':
                obj_dir = self.parse_text(node.text)
            elif node.tag == 'OBJ_NAME':
                obj_name = self.parse_entity(node.text, entities)
            elif node.tag == 'VAR':
                if current_var is not None:
                    # case where VAR does not have a PARAM
                    variables[current_var.name] = current_var
                if order_mapping is None:
                    # if no order mapping try to make one
                    name = self.parse_entity(node.text, {})
                    current_var = Term(name=name, index=count)
                    entities[name] = count
                    count += 1
                else:
                    # use order mapping if it exists
                    name = self.parse_entity(node.text, entities)
                    current_var = Term(name=name, index=entities[name])
            elif node.tag == 'PARAM':
                current_var.value = self.parse_number(node.text)
                variables[current_var.name] = current_var
                current_var = None

        return ObjectiveDeclaration(name=obj_name, direction=obj_dir, terms=variables, entities=entities)

    def parse_constraint(self, root: ET.Element, entities: dict) -> ConstraintDeclaration:
        const_dir = ''
        limit = ''
        const_type = ''
        operator = ''
        variables = OrderedDict()
        current_var = None
        for node in root:
            if node.tag == 'CONST_DIR':
                const_dir = self.parse_text(node.text)
            elif node.tag == 'OPERATOR':
                operator = self.parse_text(node.text)
            elif node.tag == 'LIMIT':
                limit = self.parse_number(node.text)
            elif node.tag == 'CONST_TYPE':
                const_type = self.parse_text(node.text)
            elif node.tag == 'VAR':
                if current_var:
                    variables[current_var.name] = current_var
                name = self.parse_entity(node.text, entities)
                current_var = Term(name=name, index=entities[name])
            elif node.tag == 'PARAM':
                current_var.value = self.parse_number(node.text)
                variables[current_var.name] = current_var
                current_var = None

        if current_var is not None and current_var.name not in variables:
            variables[current_var.name] = current_var
        if const_type == const.BALANCE_CONSTRAINT_1 or const_type == const.BALANCE_CONSTRAINT_2:
            if len(variables) != 2:
                raise ValueError(
                    f'Balance constraint has incorrect number of variables (got: {len(variables)}, expected: 2): {ET.tostring(root)}')
            if const_type == const.BALANCE_CONSTRAINT_1 and list(variables.values())[0].value is None:
                raise ValueError(
                    f'Balance constraint xby has missing value for y: {ET.tostring(root)}')
        return ConstraintDeclaration(direction=const_dir, limit=limit, operator=operator,
                                     type=const_type, terms=variables, entities=entities)

    def parse_file(self, fname: str, order_mapping=None) -> Optional[ProblemFormulation]:
        with open(fname, 'r') as fd:
            data = fd.read()
            return self.parse(data, order_mapping)


# Parses the JSON formatted training examples
class JSONFormulationParser(Parser):

    def parse(self, data: dict, order_mapping=None) -> ProblemFormulation:
        try:
            # get actual data, top level is a numeric key pointing to data
            key = ''
            for k, v in data.items():
                data = v
                key = k

            order_mapping = data['order_mapping'] if order_mapping is None else order_mapping
            objective = self.parse_objective(data['obj_declaration'], data['vars'], order_mapping)
            constraints = []
            for constraint in data['const_declarations']:
                constraints.append(self.parse_constraint(constraint, objective.entities, order_mapping))
            return ProblemFormulation(objective, constraints, order_mapping)
        except:
            logging.warning(
                f'Could not parse example {key}: {data}.\nProceeding with empty ProblemFormulation.')
            # cannot be parsed, returning an empty ProblemFormulation
            return ProblemFormulation(ObjectiveDeclaration('', {}, {}, ''), [], entities={})

    def get_data(self, data: dict):
        key = ''
        for k, v in data.items():
            data = v
            key = k
        return data

    def get_text(self, data: dict):
        return self.get_data(data)['document']

    def parse_objective(self, data: dict, vars: dict, order_mapping: dict) -> ObjectiveDeclaration:
        terms = {}

        if 'terms' in data:
            for i, (k, v) in enumerate(data['terms'].items()):
                terms[k] = Term(name=k, index=order_mapping[k], value=self.parse_number(v), value_str=v)
        else:
            # assume values are 1 if there is no terms mapping
            for var in vars:
                terms[var] = Term(name=var, index=order_mapping[var], value=1)

        return ObjectiveDeclaration(name=data['name'],
                                    direction=data['direction'],
                                    terms=terms,
                                    entities=order_mapping)

    def parse_constraint(self, data: dict, entities: dict, var_dict: dict) -> ConstraintDeclaration:
        terms = OrderedDict()
        limit_str = data['limit'] if 'limit' in data else '0'
        limit = self.parse_number(limit_str)
        constraint_type = const.TYPE_DICT[data['type']]
        direction = self.parse_text(data['direction'])
        operator = self.parse_text(data['operator'])

        if 'terms' in data:
            for k, v in data['terms'].items():
                terms[k] = Term(name=k, index=entities[k], value=self.parse_number(v), value_str=v)
        elif constraint_type == const.UPPER_BOUND or constraint_type == const.LOWER_BOUND or constraint_type == const.RATIO_CONTROL_CONSTRAINT:
            k = data['var']
            terms[k] = Term(name=k, index=entities[k])
        elif constraint_type == const.SUM_CONSTRAINT:
            # no parsing for terms needed here
            pass
        elif constraint_type == const.BALANCE_CONSTRAINT_1:
            k = data['y_var']
            terms[k] = Term(name=k, index=entities[k], value=self.parse_number(data['param']), value_str=data['param'])
            k = data['x_var']
            terms[k] = Term(name=k, index=entities[k])
        elif constraint_type == const.BALANCE_CONSTRAINT_2:
            k = data['y_var']
            terms[k] = Term(name=k, index=entities[k])
            k = data['x_var']
            terms[k] = Term(name=k, index=entities[k])
        else:
            logging.info(f'Could not find terms in {data}')

        return ConstraintDeclaration(type=constraint_type,
                                     direction=direction,
                                     operator=operator,
                                     entities=entities,
                                     limit=limit,
                                     terms=terms,
                                     limit_str=limit_str)


class RDFParser(Parser):
    def parse(self, data: object, order_mapping: Optional[dict] = None) -> ProblemFormulation:
        if isinstance(data, str):
            # slice to -1 because last index is a \n
            declarations = data.split(const.RDF_DECLARATION_SEP)[:-1]
            objective = None
            constraints = []
            exceptions = []
            for declaration in declarations:
                dec_lines = declaration.split(const.RDF_LINE_SEP)[:-1]
                dec_lines = [x.split(const.RDF_DELIMITER) for x in dec_lines]
                dec_lines = [[x.strip() for x in y] for y in dec_lines]
                old_len_lines = len(dec_lines)
                dec_lines = [x for x in dec_lines if len(x) == 3]
                new_len_lines = len(dec_lines)

                if self.is_objective(dec_lines):
                    try:
                        if old_len_lines > new_len_lines:
                            exceptions.append(ObjectiveNotTriple('Not RDF Triple'))
                        objective = self.parse_objective(dec_lines, order_mapping)
                    except Exception as e:
                        exceptions.append(e)
                else:
                    try:
                        if old_len_lines > new_len_lines:
                            exceptions.append(ConstraintNotTriple('Not RDF Triple'))
                        constraints.append(self.parse_constraint(dec_lines, order_mapping))
                    except Exception as e:
                        exceptions.append(e)

            return ProblemFormulation(objective, constraints, order_mapping, exceptions)

    def parse_objective(self, data: list[list[str]], order_mapping: dict):
        objective_name = ''
        objective_direction = ''
        terms = {}
        for line in data:
            if line[1] == const.RDF_OBJ_TYPE:
                objective_name = line[0]
                objective_direction = line[2]
            else:
                term_name = line[2]
                term_value = line[1]
                if term_value == '':
                    term_value = '1'
                term_name = self.parse_entity(term_name, order_mapping)
                terms[term_name] = Term(
                    term_name,
                    order_mapping[term_name],
                    self.parse_number(term_value),
                    term_value
                )
        return ObjectiveDeclaration(objective_direction, terms, order_mapping, objective_name)

    def parse_constraint(self, data: list[list[str]], order_mapping: dict):
        constraint_direction = ''
        constraint_type = ''
        constraint_limit = ''
        operator = ''
        terms = None

        for line in data:
            if line[1] == const.RDF_CONST_TYPE:
                constraint_type = line[2]
                constraint_direction = line[0]
            elif line[1] == const.RDF_CONST_LIM:
                constraint_limit = line[2]
            elif line[1] == const.RDF_OPERATOR:
                operator = line[2]

        if constraint_type == const.SUM_CONSTRAINT:
            terms = OrderedDict()
        elif constraint_type == const.LOWER_BOUND or constraint_type == const.UPPER_BOUND:
            terms = self.parse_constraint_terms(data, order_mapping, None)
        elif constraint_type == const.LINEAR_CONSTRAINT:
            terms = self.parse_constraint_terms(data, order_mapping, 1)
        elif constraint_type == const.RATIO_CONTROL_CONSTRAINT:
            terms = self.parse_constraint_terms(data, order_mapping, 1)
        elif constraint_type == const.BALANCE_CONSTRAINT_1 or constraint_type == const.BALANCE_CONSTRAINT_2:
            terms = self.parse_constraint_terms_xby(data, order_mapping)
        else:
            raise UnidentifiedDeclarationException(f'Constraint type {constraint_type} not valid')

        return ConstraintDeclaration(direction=constraint_direction,
                                     terms=terms,
                                     entities=order_mapping,
                                     type=constraint_type,
                                     operator=operator,
                                     limit=self.parse_number(constraint_limit),
                                     limit_str=constraint_limit)

    def parse_constraint_terms(self, data, order_mapping, value_idx=None):
        terms = OrderedDict()
        for line in data:
            if line[1] in [const.RDF_CONST_TYPE, const.RDF_CONST_LIM, const.RDF_OPERATOR]:
                continue
            else:
                term_name = line[2]
                if value_idx is not None:
                    term_value = line[1]
                    if term_value == const.RDF_VAR:
                        term_value = '1'
                    parsed_value = self.parse_number(term_value)
                else:
                    term_value = ''
                    parsed_value = None

                term_name = self.parse_entity(term_name, order_mapping)
                terms[term_name] = Term(
                    term_name,
                    order_mapping[term_name],
                    parsed_value,
                    term_value
                )
        return terms

    def parse_constraint_terms_xby(self, data, order_mapping):
        terms = OrderedDict()
        for line in data:
            if line[1] in [const.RDF_CONST_TYPE, const.RDF_CONST_LIM, const.RDF_OPERATOR]:
                continue
            else:
                term_name = line[2]
                term_value = line[1]
                if term_value == 'var':
                    term_value = '1'
                if term_value == const.RDF_XBY_Y:
                    term_name = self.parse_entity(term_name, order_mapping)
                    terms[term_name] = Term(
                        term_name,
                        order_mapping[term_name]
                    )
                    terms.move_to_end(term_name, last=False)
                elif term_value == const.RDF_XBY_X:
                    term_name = self.parse_entity(term_name, order_mapping)
                    terms[term_name] = Term(
                        term_name,
                        order_mapping[term_name]
                    )
                else:
                    term_name = self.parse_entity(term_name, order_mapping)
                    terms[term_name] = Term(
                        term_name,
                        order_mapping[term_name],
                        self.parse_number(term_value),
                        term_value
                    )
                    terms.move_to_end(term_name, last=False)
        return terms

    def is_constraint(self, data: list[list[str]]):
        for line in data:
            if line[1] == const.RDF_CONST_TYPE:
                return True
            elif line[1] == const.RDF_OBJ_TYPE:
                return False

        raise UnidentifiedDeclarationException("Declaration is neither a constraint nor objective")

    def is_objective(self, data: list[list[str]]):
        return not self.is_constraint(data)

    def find_direction(self, data: list[list[str]]):
        return data[0][0]


class LimitNotFoundException(Exception):
    pass


class WrongDeclarationTypeException(Exception):
    pass


class UnidentifiedDeclarationException(Exception):
    pass


class SyntaxErrorException(Exception):
    pass


class ObjectiveNotTriple(Exception):
    pass


class ConstraintNotTriple(Exception):
    pass
