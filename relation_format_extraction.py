from typing import List
from spacy.language import Doc, Language, Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
import spacy
from spacy.attrs import LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY
from random import shuffle


def clean_list_string(str_list: str):
    str_list = str_list.replace('[', '')
    str_list = str_list.replace(']', '')
    str_list = str_list.replace('\'', '')
    str_list = str_list.replace('\n', '')
    return str_list


class NERToken:
    def __init__(self, word: str, label: str):
        self.word = word
        self.label = label

    def __str__(self):
        return self.word + ' ' + self.label


class Sentence:
    def __init__(self, id: int, token_with_labels: List[NERToken], token_with_predictions: List[NERToken] = None):
        self.id = id
        self.tokens = token_with_labels
        self.label_dict = Sentence.dict_building(token_with_labels)
        self.prediction_dict = Sentence.dict_building(token_with_predictions)

    def dict_building(ner_tokens: List[NERToken]):
        if ner_tokens is None:
            return None
        labels = [t.label for t in ner_tokens]
        drug_starts = [i for i in range(len(labels)) if labels[i].startswith('B')]
        drug_dict = dict()
        for index in range(len(drug_starts)):
            i = drug_starts[index]
            j = i+1
            while j < len(labels):
                if labels[j] == 'O' or labels[j].startswith('B'):
                    break
                else:
                    j += 1
            drug_tokens = (i, j)
            drug_dict.__setitem__(index, drug_tokens)
        return drug_dict


class Pair:
    def __init__(self, e1_id: int, e2_id: int, e1_index: int, e2_index: int, e1_text: str, e2_text: str):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.e1_index = e1_index
        self.e2_index = e2_index
        self.e1_text = e1_text
        self.e2_text = e2_text


class Instance():
        def __init__(self, doc: Doc, original_doc: Doc, sentence_id: int, pair: Pair):
            self.doc = doc
            self.original_doc = original_doc
            self.sentence_id = sentence_id
            self.e1_id = pair.e1_id
            self.e2_id = pair.e2_id
            self.pair_id = str(self.sentence_id) + '.' + str(self.e1_id)+'.'+str(self.e2_id)

        def __str__(self):
            return self.doc.text


def get_tokenized_sentences(sentences_path, labels_path):
    sentence_file = open(sentences_path, 'r').readlines()
    predictions = open(labels_path, 'r').readlines()
    assert len(sentence_file) == len(predictions)
    sentences = list()
    for i in range(len(sentence_file)):
        sentence = sentence_file[i]
        prediction = predictions[i]
        sentence = clean_list_string(sentence)
        prediction = clean_list_string(prediction)
        words = sentence.split(', ')
        labels = prediction.split(', ')
        assert len(words) == len(labels)
        tokens = [NERToken(words[j], labels[j]) for j in range(len(words)) if words[j] != '']
        ner_sentence = Sentence(i, tokens)
        sentences.append(ner_sentence)
    return sentences


def substitution(doc: Doc, pair: Pair, drugs) -> Doc:
    index_1 = pair.e1_index
    index_2 = pair.e2_index
    name_1 = pair.e1_text
    name_2 = pair.e2_text
    np_array = doc.to_array([LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY])
    word_list = [t.text for i, t in enumerate(doc)]
    no_pair = False
    if name_1.text.lower() == name_2.text.lower():
        word_list[index_1] = 'NoPair'
        word_list[index_2] = 'NoPair'
        no_pair = True
    for index, name in drugs:
        if index != index_1 and index != index_2:
            word_list[index] = 'Drug'
    if not no_pair:
        word_list[index_1] = 'PairDrug1'
        word_list[index_2] = 'PairDrug2'
    doc2 = Doc(doc.vocab, words=word_list)
    doc2.from_array([LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY], np_array)
    return doc2


def custom_tokenizer(nlp):
    prefix_re = compile_prefix_regex(Language.Defaults.prefixes + (';', '\*'))
    suffix_re = compile_suffix_regex(Language.Defaults.suffixes + (';', '\*'))
    infix_re = compile_infix_regex(Language.Defaults.infixes + ('(', ')', "/", "-", ";", "\*"))
    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)


nlp = spacy.load('en')
nlp.tokenizer = custom_tokenizer(nlp)


def get_sentences_with_ids(sentences_path: str, predictions_path: str, labels_file: str):
    sentence_file = open(sentences_path, 'r').readlines()
    predictions = open(predictions_path, 'r').readlines()
    labels = open(labels_file, 'r').readlines()
    assert len(sentence_file) == len(predictions) == len(labels)
    sentences = list()
    for i in range(len(sentence_file)):
        sentence = sentence_file[i]
        prediction = predictions[i]
        label = clean_list_string(labels[i])
        sentence = clean_list_string(sentence)
        prediction = clean_list_string(prediction)
        words = sentence.split(', ')
        prediction_list = prediction.split(', ')
        label_list = label.split(', ')
        assert len(prediction_list) == len(words) == len(label_list)
        token_with_labels = [NERToken(words[j], label_list[j]) for j in range(len(words)) if words[j] != '']
        token_with_predictions = [NERToken(words[j], prediction_list[j]) for j in range(len(words)) if words[j] != '']
        ner_sentence = Sentence(i, token_with_labels, token_with_predictions)
        sentences.append(ner_sentence)


def generate_instances() -> List[Instance]:
    instances = list()
    sentences = get_tokenized_sentences('DDI_NER_inputSent.txt', 'DDI_NER_predictLabels.txt')
    shuffle(sentences)
    infixes = ['(', ')', '/', '-', ';', '*']
    s_id = 0
    for s in sentences:
        tokens = s.tokens
        drug_keys = list(s.label_dict.keys())
        spaces = list()
        for i in range(len(tokens)-1):
            actual = tokens[i]
            next = tokens[i+1]
            if actual.word in infixes or next.word in infixes:
                space = False
            else:
                space = True
            spaces.append(space)
        spaces.append(False)
        words = [t.word for t in tokens]
        assert len(spaces) == len(words)
        doc = Doc(nlp.vocab, words, spaces)
        merged = False
        for j in range(len(drug_keys)):
            start, end = s.label_dict.__getitem__(drug_keys[j])
            length = end - start
            if length > 1:
                # print(list(doc))
                # span = doc[start:end]
                with doc.retokenize() as retokenizer:
                    retokenizer.merge(doc[start:end])
                for k in range(j+1, len(drug_keys)):
                    n_start, n_end = s.label_dict.__getitem__(drug_keys[k])
                    n_start -= length-1
                    n_end -= length-1
                    s.label_dict.__setitem__(k, (n_start, n_end))
        pairs = list()
        drug_indexes = [s.label_dict.get(drug_keys[i]) for i in range(len(drug_keys))]
        drugs = [(i, doc[i]) for (i, end) in drug_indexes]
        if len(drugs) >= 2:
            for i in range(len(drug_keys)-1):
                for j in range(i+1, len(drug_keys)):
                    drug_i, end_1 = s.label_dict.get(drug_keys[i])
                    drug_j, end_2 = s.label_dict.get(drug_keys[j])
                    i_text = doc[drug_i]
                    j_text = doc[drug_j]
                    p = Pair(drug_keys[i], drug_keys[j], drug_i, drug_j, i_text, j_text)
                    pairs.append(p)
        for p in pairs:
            new_doc = substitution(doc, p, drugs)
            instance = Instance(new_doc, doc, s_id, p)
            instances.append(instance)
        s_id += 1
    return instances


get_sentences_with_ids('DDI_NER_inputSent.txt', 'DDI_NER_predictLabels.txt', 'DDI_NER_correctLabels.txt')
instances = generate_instances()
for instance in instances:
    print(instance.doc)