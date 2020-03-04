from typing import List
from spacy.language import Doc
from constants import *


class NERToken:
    def __init__(self, word: str, label: str):
        self.word = word
        self.label = label

    def __str__(self):
        return self.word + ' ' + self.label

    def __len__(self):
        return len(self.word)


class CompleteNERToken:
    def __init__(self, word: str, label: str, id: str):
        self.word = word
        self.label = label
        self.id = id

    def __str__(self):
        return self.word + ' ' + self.label

    def __len__(self):
        return len(self.word)


class SequencePair:

    def __init__(self, first_sequence, second_sequence):
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        assert len(self.first_sequence) == len(self.second_sequence)
        self.word_list = [token.word for token in self.first_sequence]
        self.first_labels = [token.label for token in self.first_sequence]
        self.second_labels = [token.label for token in self.second_sequence]
        self.doc = self.create_doc()

    def __len__(self):
        return len(self.first_sequence)

    def create_doc(self):
        tokens = self.word_list
        infixes = ['(', ')', '/', '-', ';', '*']
        spaces = list()
        for i in range(len(tokens) - 1):
            actual = tokens[i]
            next = tokens[i + 1]
            if actual in infixes or next in infixes:
                space = False
            else:
                space = True
            spaces.append(space)
        spaces.append(False)
        try:
            doc = Doc(nlp.vocab, tokens, spaces)
        except ValueError:
            doc = Doc(nlp.vocab, tokens)
        return doc


class Interval:
    def __init__(self, a: int, b: int):
        self.low = a
        self.high = b

    def __str__(self):
        return str(self.low) + '-' + str(self.high)


class PairWithInterval:
    def __init__(self, head_id: str, head_interval: Interval, tail_id: str, tail_interval: Interval):
        self.head_id = head_id
        self.head_interval = head_interval
        self.tail_id = tail_id
        self.tail_interval = tail_interval

    def __str__(self):
        return self.head_id + '(' + str(self.head_interval) + '), ' + self.tail_id + '(' + str(self.tail_interval) + ')'


class Sentence:
    def __init__(self, id: str, token_with_labels: List[NERToken] = None,
                 token_with_predictions: List[NERToken] = None):
        self.id = id
        self.original_tokens = token_with_labels
        self.predicted_tokens = token_with_predictions
        self.label_dict = Sentence.dict_building(token_with_labels, with_id=True)
        self.prediction_list = Sentence.dict_building(token_with_predictions, with_id=False)
        self.correct_drugs: dict = None
        self.approximate_drugs: dict = None
        self.merged_drug_starts = None
        self.wrong_drugs: dict = None
        self.missing_drugs: dict = None
        self.doc = None
        self.complete_list = None
        self.count_approximate = True

    def __str__(self):
        if self.doc is None:
            return str(self.original_tokens)
        return str(self.doc)

    def __len__(self):
        return len(self.original_tokens)

    def set_predictions(self, token_with_predictions: List[NERToken]):
        self.predicted_tokens = token_with_predictions
        self.prediction_list = Sentence.dict_building(token_with_predictions, with_id=False)
        self.check_correct()

    def dict_building(ner_tokens: List[NERToken], with_id: bool):
        if ner_tokens is None:
            return None
        labels = [t.label for t in ner_tokens]
        drug_starts = [i for i in range(len(labels)) if labels[i].startswith('B')]
        drug_dict = dict()
        drug_list = list()
        for index in range(len(drug_starts)):
            i = drug_starts[index]
            id = labels[i].replace('B-', '')
            j = i + 1
            while j < len(labels):
                if labels[j] == 'O' or labels[j].startswith('B'):
                    break
                else:
                    j += 1
            drug_tokens = (i, j)
            if with_id:
                drug_dict.__setitem__(id, drug_tokens)
            else:
                drug_list.append(drug_tokens)
        if with_id:
            return drug_dict
        else:
            return drug_list

    def check_correct(self):
        self.correct_drugs = dict()
        self.approximate_drugs = dict()
        self.wrong_drugs = dict()
        self.missing_drugs = dict()
        wrong_index = 0
        for k in self.label_dict.keys():
            correct_start, correct_end = self.label_dict.get(k)
            for start, end in self.prediction_list:
                if abs(end - correct_end) <= 5:
                    original_string = ''
                    predicted_string = ''
                    for i in range(correct_start, correct_end):
                        original_string += self.original_tokens[i].word
                    for i in range(start, end):
                        predicted_string += self.predicted_tokens[i].word
                    if original_string == predicted_string:
                        if (start, end) not in self.correct_drugs.values():
                            self.correct_drugs.__setitem__(k, (start, end))
                            break
                    elif predicted_string in original_string:
                        if len(predicted_string) >= 0.5 * len(original_string) and self.count_approximate:
                            self.approximate_drugs.__setitem__(k, (start, end))
        for (start, end) in self.prediction_list:
            if (start, end) not in self.correct_drugs.values() and (start, end) not in self.approximate_drugs.values():
                wrong_id = self.id + '.w' + str(wrong_index)
                self.wrong_drugs.__setitem__(wrong_id, (start, end))
                wrong_index += 1
        for k in self.label_dict:
            if k not in self.correct_drugs.keys() and k not in self.approximate_drugs.keys():
                self.missing_drugs.__setitem__(k, self.label_dict.get(k))
        complete_list = list()
        for k in self.correct_drugs.keys():
            start, end = self.correct_drugs.get(k)
            complete_list.append((start, end, k, 'C'))
        for k in self.approximate_drugs.keys():
            start, end = self.approximate_drugs.get(k)
            complete_list.append((start, end, k, 'A'))
        for k in self.wrong_drugs.keys():
            start, end = self.wrong_drugs.get(k)
            complete_list.append((start, end, k, 'W'))
        complete_list = sorted(complete_list)
        print(complete_list)
        self.complete_list = complete_list

    def generate_pairs(self):
        pairs = list()
        for i in range(len(self.label_dict.keys()) - 1):
            head_key = list(self.label_dict.keys())[i]
            head_a, head_b = self.label_dict.get(head_key)
            head_interval = Interval(head_a, head_b)
            for j in range(i + 1, len(self.label_dict.keys())):
                tail_key = list(self.label_dict.keys())[j]
                tail_a, tail_b = self.label_dict.get(tail_key)
                tail_interval = Interval(tail_a, tail_b)
                pair = PairWithInterval(head_key, head_interval, tail_key, tail_interval)
                pairs.append(pair)
        return pairs

    def no_substitution_doc(self) -> Doc:
        tokens = self.original_tokens
        infixes = ['(', ')', '/', '-', ';', '*']
        spaces = list()
        for i in range(len(tokens) - 1):
            actual = tokens[i]
            next = tokens[i + 1]
            if actual.word in infixes or next.word in infixes:
                space = False
            else:
                space = True
            spaces.append(space)
        spaces.append(False)
        words = [t.word for t in tokens]
        try:
            doc = Doc(nlp.vocab, words, spaces)
        except ValueError:
            doc = Doc(nlp.vocab, words)
        self.doc = doc
        return doc

class Pair:
    def __init__(self, e1_id: str, e2_id: str, e1_index: int, e2_index: int, e1_text: str, e2_text: str,
                 sentence: Sentence):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.e1_index = e1_index
        self.e2_index = e2_index
        self.e1_text = e1_text
        self.e2_text = e2_text
        self.type = None
        self.sentence = sentence

    def set_type(self, type: str):
        self.type = type


class JointInstance:
    def __init__(self, doc: Doc, original_doc: Doc, pair: Pair):
        self.doc = doc
        self.original_doc = original_doc
        self.e1_id = pair.e1_id
        self.e2_id = pair.e2_id
        self.pair = pair
        self.class_value = ''
        self.type = self.pair.type
        self.dependency_path = None

    def __str__(self):
        return self.doc.text

    def set_dependency_path(self, dependency_path):
        self.dependency_path = dependency_path

    def __len__(self):
        return len(self.doc)