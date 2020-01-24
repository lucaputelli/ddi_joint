from typing import List
from spacy.language import Doc, Language, Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
import spacy
from spacy.attrs import LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY
from random import shuffle
from pre_processing_lib import get_sentences


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
    def __init__(self, id: str, token_with_labels: List[NERToken] = None, token_with_predictions: List[NERToken] = None):
        self.id = id
        self.original_tokens = token_with_labels
        self.predicted_tokens = token_with_predictions
        self.label_dict = Sentence.dict_building(token_with_labels, with_id=True)
        self.prediction_list = Sentence.dict_building(token_with_predictions, with_id=False)
        self.correct_drugs = None
        self.merged_drug_starts = None
        self.doc = None

    def __str__(self):
        if self.doc is None:
            return str(self.original_tokens)
        return str(self.doc)

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
            j = i+1
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
                        print(k, start, end, correct_start, correct_end)
                        if (start, end) not in self.correct_drugs.values():
                            self.correct_drugs.__setitem__(k, (start, end))
                            break

class Pair:
    def __init__(self, e1_id: str, e2_id: str, e1_index: int, e2_index: int, e1_text: str, e2_text: str):
        self.e1_id = e1_id
        self.e2_id = e2_id
        self.e1_index = e1_index
        self.e2_index = e2_index
        self.e1_text = e1_text
        self.e2_text = e2_text


class Instance():
        def __init__(self, doc: Doc, original_doc: Doc, pair: Pair):
            self.doc = doc
            self.original_doc = original_doc
            self.e1_id = pair.e1_id
            self.e2_id = pair.e2_id
            self.pair = pair
            self.class_value = ''
            # self.pair_id = str(self.sentence_id) + '.' + str(self.e1_id)+'.'+str(self.e2_id)

        def __str__(self):
            return self.doc.text

        def set_class(self, class_value: str):
            self.class_value = class_value


def get_tokenized_sentences(sentences_path, labels_path):
    sentence_file = open(sentences_path, 'r').readlines()
    predictions = open(labels_path, 'r').readlines()
    assert len(sentence_file) == len(predictions)
    sentences = list()
    for i in range(len(sentence_file)):
        id = sentence_file[i].split(': [')[0].replace(';', '')
        sentence = sentence_file[i].split(': [')[1]
        sentence = clean_list_string(sentence)
        id_pred = predictions[i].split(': [')[0].replace(';', '')
        prediction = predictions[i].split(': [')[1]
        words = sentence.split(', ')
        labels = prediction.split(', ')
        labels = [clean_list_string(label) for label in labels]
        assert len(words) == len(labels)
        assert id == id_pred
        tokens = [NERToken(words[j], labels[j]) for j in range(len(words)) if words[j] != '']
        if not tokens:
            print(sentence_file[i])
        ner_sentence = Sentence(id, tokens)
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


def generate_gold_standard(sentences: List[Sentence]) -> (List[Instance], List[Sentence]):
    instances = list()
    infixes = ['(', ')', '/', '-', ';', '*']
    for s in sentences:
        id = s.id
        tokens = s.original_tokens
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
        try:
            doc = Doc(nlp.vocab, words, spaces)
        except ValueError:
            doc = Doc(nlp.vocab, words)
        for j in range(len(drug_keys)):
            start, end = s.label_dict.__getitem__(drug_keys[j])
            length = end - start
            if length > 1:
                # print(list(doc))
                # span = doc[start:end]
                with doc.retokenize() as retokenizer:
                    try:
                        retokenizer.merge(doc[start:end])
                    except IndexError:
                        print(words)
                for k in range(j+1, len(drug_keys)):
                    tokens = list(doc)
                    key = drug_keys[k]
                    n_start, n_end = s.label_dict.__getitem__(key)
                    n_start -= length-1
                    n_end -= length-1
                    s.label_dict.__setitem__(key, (n_start, n_end))
        pairs = list()
        drug_indexes = [s.label_dict.get(drug_keys[i]) for i in range(len(drug_keys))]
        s.merged_drug_starts = [i for (i, end) in drug_indexes]
        s.doc = doc
        try:
            drugs = [(i, doc[i]) for (i, end) in drug_indexes]
        except IndexError:
            print(drug_indexes, words)
        if len(drugs) >= 2:
            for i in range(len(drug_keys)-1):
                for j in range(i+1, len(drug_keys)):
                    drug_i, end_1 = s.label_dict.get(drug_keys[i])
                    drug_j, end_2 = s.label_dict.get(drug_keys[j])
                    try:
                        i_text = doc[drug_i]
                    except IndexError:
                        i_text = 'NoWord'
                    j_text = doc[drug_j]
                    p = Pair(drug_keys[i], drug_keys[j], drug_i, drug_j, i_text, j_text)
                    pairs.append(p)
        for p in pairs:
            new_doc = substitution(doc, p, drugs)
            instance = Instance(new_doc, doc, p)
            instances.append(instance)
    xml_pairs = get_pairs_from_xml()
    for i in range(len(instances)):
        print(i)
        p = instances[i].pair
        e1_id = clean_list_string(p.e1_id)
        e2_id = clean_list_string(p.e2_id)
        found = False
        for (e1, e2, class_value) in xml_pairs:
            if e1 == e1_id and e2 == e2_id:
                found = True
                instances[i].set_class(class_value)
                break
    return instances, sentences


def get_pairs_from_xml():
    xml_sentences = get_sentences('Dataset/Test/Overall')
    pairs = list()
    for i in range(0, len(xml_sentences)):
        # s_id = xml_sentences[i].attributes['id'].value
        s_pairs = xml_sentences[i].getElementsByTagName('pair')
        for s_pair in s_pairs:
            e1 = s_pair.attributes['e1'].value
            e2 = s_pair.attributes['e2'].value
            ddi = s_pair.attributes['ddi'].value
            if ddi == 'false':
                class_value = 'unrelated'
            else:
                class_value = s_pair.attributes['type'].value
            pairs.append((e1, e2, class_value))
    return pairs


def check_ids():
    sentences = get_tokenized_sentences('DDI_Test_Sent_Gold.txt', 'DDI_Test_IOB2_Gold.txt')
    sentences = [s for s in sentences if s.original_tokens]
    # sentences.sort(key=lambda x: x.id)
    xml_sentences = get_sentences('Dataset/Test/Overall')
    wrong = 0
    wrong_list = []
    for i in range(0, len(xml_sentences)):
        entities = xml_sentences[i].getElementsByTagName('entity')
        s_id = xml_sentences[i].attributes['id'].value
        id = sentences[i].id
        text = xml_sentences[i].attributes['text'].value
        words = [token.word for token in sentences[i].original_tokens]
        xml_entity_number = len(entities)
        entity_number = len(sentences[i].label_dict.keys())
        print(text, words)
        if xml_entity_number != entity_number:
            wrong += 1
            wrong_list.append((s_id, xml_entity_number, entity_number))
    print(wrong)
    print(wrong_list)


def sentences_from_prediction(sentences_path, labels_path):
    gold_sentences = get_tokenized_sentences('MANUALLY_CHECKED_TOKEN.txt', 'MANUALLY_CHECKED_ID.txt')
    sentences_dict = {s.id: s for s in gold_sentences}
    sentence_file = open(sentences_path, 'r').readlines()
    predictions = open(labels_path, 'r').readlines()
    assert len(sentence_file) == len(predictions)
    sentences = list()
    wrong = list()
    for i in range(len(sentence_file)):
        id = sentence_file[i].split(': [')[0].replace('; ', '')
        sentence = sentence_file[i].split(': [')[1]
        sentence = clean_list_string(sentence)
        id_pred = predictions[i].split(': [')[0].replace('; ', '')
        prediction = clean_list_string(predictions[i].split(': [')[1])
        words = sentence.split(', ')
        labels = prediction.split(', ')
        labels = [clean_list_string(label) for label in labels]
        if len(words) != len(labels):
            wrong.append(id)
            continue
        # assert id == id_pred
        length = min(len(words), len(labels))
        tokens = [NERToken(words[j], labels[j]) for j in range(length) if words[j] != '']
        if not tokens:
            print(sentence_file[i])
        ner_sentence = sentences_dict.get(id_pred)
        ner_sentence.set_predictions(tokens)
        sentences.append(ner_sentence)
    infixes = ['(', ')', '/', '-', ';', '*']
    for s in sentences:
        id = s.id
        tokens = s.tokens
        drug_indexes = s.prediction_list
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
        for j in range(len(drug_indexes)):
            start, end = drug_indexes[j]
            length = end - start
            if length > 1:
                # print(list(doc))
                # span = doc[start:end]
                with doc.retokenize() as retokenizer:
                    try:
                        retokenizer.merge(doc[start:end])
                    except IndexError:
                        print(words)
                for k in range(j + 1, len(drug_indexes)):
                    tokens = list(doc)
                    n_start, n_end = drug_indexes[k]
                    n_start -= length - 1
                    n_end -= length - 1
                    s.prediction_list[k] = (n_start, n_end)
        s.merged_drug_starts = [start for (start, end) in s.prediction_list]
        s.doc = doc
    return sentences


sentences = sentences_from_prediction('inputSent2.txt', 'predLabels2_modified.txt')