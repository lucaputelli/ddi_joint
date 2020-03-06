from xml.dom.minidom import Element
from xml.dom import minidom
from spacy.tokens import Doc
import spacy
from spacy.language import Language, Tokenizer
from spacy.attrs import LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY
from os import listdir
import en_core_web_sm
import numpy as np
import re
from typing import Iterable, List
from spacy import displacy
from pathlib import Path
import networkx as nx
from networkx.exception import *


class Instance:

    def __init__(self, sentence: Element, pair: Element, doc: Doc):
        self._sentence = sentence
        self._pair = pair
        self._doc = doc
        self._dependency_path = None
        self._depth_first = None
        self._breadth_first = None

    def get_text(self) -> str:
        return self._sentence.attributes['text'].value

    def get_sentence(self) -> Element:
        return self._sentence

    def ged_pair(self) -> Element:
        return self._pair

    def get_doc(self) -> Doc:
        return self._doc

    def get_e1(self) -> str:
        return self._pair.attibutes['e1'].value

    def get_e2(self) -> str:
        return self._pair.attributes['e2'].value

    def get_sentence_id(self) -> str:
        return self._sentence.attributes['id'].value

    def get_pair_id(self) -> str:
        return self._pair.attributes['id'].value

    def get_class(self) -> str:
        ddi = self._pair.attributes['ddi'].value
        if ddi == 'true':
            if self._pair.hasAttribute('type'):
                return self._pair.attributes['type'].value
            else:
                #print(self.get_pair_id())
                return 'int'
        else:
            return ddi

    def get_dependency_path(self) -> Iterable:
        return self._dependency_path

    def set_dependency_path(self, dependency_path: Iterable):
        self._dependency_path = dependency_path

    def set_breadth_first(self, breadth_first: Iterable):
        self._breadth_first = breadth_first

    def set_depth_first(self, depth_first: Iterable):
        self._depth_first = depth_first

    def get_depth_first(self) -> Iterable:
        return self._depth_first

    def get_breadth_first(self) -> Iterable:
        return self._breadth_first


nlp = en_core_web_sm.load()
prefix_re = spacy.util.compile_prefix_regex(Language.Defaults.prefixes)
suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes)
infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes + ("/", "-", ";"))


def is_discontinuous(entity: Element) -> bool:
    offset_string = entity.attributes['charOffset'].value
    interval_number = str.count(offset_string, '-')
    if interval_number > 1:
        return True
    else:
        return False


def is_braided(sentence: Element) -> bool:
    text = sentence.attributes['text'].value
    #print(text)
    intervals = list()
    entities = sentence.getElementsByTagName('entity')
    discontinuous = False
    for j in range(len(entities)):
        discontinuous = discontinuous or is_discontinuous(entities[j])
    if not discontinuous:
        return False
    for j in range(len(entities)):
        offset_string = entities[j].attributes['charOffset'].value
        interval_number = str.count(offset_string, '-')
        if interval_number == 1:
            offsets = str.split(offset_string, "-")
            left = int(offsets.__getitem__(0))
            right = int(offsets.__getitem__(1))
            intervals += [(left, right)]
        else:
            split = str.split(offset_string, ';')
            for s in split:
                offsets = str.split(s, "-")
                left = int(offsets.__getitem__(0))
                right = int(offsets.__getitem__(1))
                intervals += [(left, right)]
    for i1 in intervals:
        for i2 in intervals:
            if i1 != i2:
                left1 = i1[0]
                right1 = i1[1]
                left2 = i2[0]
                right2 = i2[1]
                if left2 >= left1 and right2 <= right1:
                    #print(i1, i2)
                    return True
                if left1 >= left2 and right1 <= right2:
                    #print(i1, i2)
                    return True
    return False


def substitution(doc: Doc, index: int, value: int) -> Doc:
    np_array = doc.to_array([LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY])
    words = [t.text for i, t in enumerate(doc)]
    #print(words[index])
    if value == -1:
        item = 'NoPair'
    if value == 0:
        item = "Drug"
    if value == 1:
        item = "PairDrug1"
    if value == 2:
        item = "PairDrug2"
    words.__setitem__(index, item)
    doc2 = Doc(doc.vocab, words=words)
    doc2.from_array([LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY], np_array)
    return doc2


def doc_cleaning(doc: Doc):
    np_array = doc.to_array([LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY])
    words = [t.text for i, t in enumerate(doc)]
    cleaned_words = list()
    for w in words:
        if w != 'PairDrug1' and w != 'PairDrug2':
            w = number_substitution(w)
        if w == '%':
            w = ' '
        cleaned_words.append(w)
    doc2 = Doc(doc.vocab, words=cleaned_words)
    doc2.from_array([LEMMA, LOWER, POS, TAG, ENT_TYPE, IS_ALPHA, DEP, HEAD, SPACY], np_array)
    return doc2


def get_sentences(path: str) -> List[Element]:
    files = listdir(path)
    tot_sentences = []
    for f in files:
        doc = minidom.parse(path + "/" + f.title())
        sentences = doc.getElementsByTagName('sentence')
        tot_sentences += sentences
    return tot_sentences


def number_substitution(text: str) -> str:
    return re.sub(r'[0-9]+[.0-9]*%?', 'NUM', text)


# Performs tokenization, detokenization and substitution
def get_instances(sentences) -> List[Instance]:
    instances = list()
    for i in range(len(sentences)):
        braided = is_braided(sentences[i])
        # if braided:
        #    continue
        entities = sentences[i].getElementsByTagName('entity')
        entity_tuples = list()
        text = str(sentences[i].attributes['text'].value)
        nlp_doc = nlp(text)
        tokens = list(nlp_doc)
        # displacy.serve(nlp_doc, style='dep')
        for j in range(len(entities)):
            e_text = entities[j].attributes['text'].value
            offset_string = entities[j].attributes['charOffset'].value
            split = str.split(offset_string, ';')
            if len(split) > 1:
                continue
            for s in split:
                offsets = str.split(s, "-")
                left = int(offsets.__getitem__(0))
                right = int(offsets.__getitem__(1))
                entity = (e_text, left, right)
                entity_tuples += [entity]
        # print(entity_tuples)
        for entity in entity_tuples:
            left_tuple = entity[1]
            right_tuple = entity[2]
            for k in range(len(tokens)):
                t = tokens.__getitem__(k)
                left_idx = t.idx
                length = len(t.text)
                right_idx = t.idx + length - 1
                if left_tuple == left_idx:
                    if right_idx == right_tuple:
                        a = 0
                        # print(t)
                    else:
                        n = 1
                        # print(right_tuple, right_idx)
                        while right_idx < right_tuple:
                            if k + n >= len(tokens):
                                break
                            next = tokens.__getitem__(k + n)
                            right_idx = next.idx + len(next.text)
                            n = n + 1
                        if (right_idx - 1) >= right_tuple:
                            span = nlp_doc[k: k + n]
                            span.merge()
                            # print(tokens[k:k + n])
        tokens = nlp_doc
        ents = sentences[i].getElementsByTagName('entity')
        pairs = sentences[i].getElementsByTagName('pair')
        for pair_index in range(len(pairs)):
            new_doc = tokens
            pair = pairs[pair_index]
            e1 = pair.attributes['e1'].value
            e2 = pair.attributes['e2'].value
            entity_triples = list()
            for l in range(len(ents)):
                etext = ents[l].attributes['text'].value
                offset = ents[l].attributes['charOffset'].value
                left = int(offset.split("-")[0])
                index = -1
                for m in range(len(tokens)):
                    token = tokens[m]
                    l_idx = token.idx
                    if etext in token.text and left == l_idx:
                        index = m
                        break
                e_id = ents[l].attributes['id'].value
                entity_triples += [(e_id, etext, index)]
            # print(entity_triples)
            for (ent_id, text, sub_index) in entity_triples:
                if ent_id == e1:
                    text1 = text
                    sub_index1 = sub_index
                    my_doc = substitution(new_doc, sub_index, 1)
                    #print(my_doc)
                    break
            for (ent_id, text, sub_index) in entity_triples:
                if ent_id == e2:
                    text2 = text
                    sub_index2 = sub_index
                    my_doc = substitution(my_doc, sub_index, 2)
                    #print(my_doc)
                    break
            for (ent_id, text, sub_index) in entity_triples:
                if ent_id != e1 and ent_id != e2:
                    my_doc = substitution(my_doc, sub_index, 0)
                    #print(my_doc)
            if text1.lower() == text2.lower():
                my_doc = substitution(my_doc, sub_index1, -1)
                my_doc = substitution(my_doc, sub_index2, -1)
            my_doc = doc_cleaning(my_doc)
            print(my_doc)
            instance = Instance(sentences[i], pair, my_doc)
            if instance not in instances:
                instances += [instance]
    return instances


def graph_creation(doc: Doc) -> nx.Graph:
    edges = []
    for token in doc:
        ancestors = list(token.ancestors)
        if len(ancestors) == 0:
            root = token.lower_ + '-' + str(token.i)
        for child in token.children:
            child_dep = child.dep_
            edges.append(('{0}-{1}'.format(token.lower_, token.i),
                          '{0}-{1}'.format(child.lower_, child.i), child_dep))
    my_graph = nx.Graph()
    # ATTENZIONE: se si crea un albero diretto diGraph() NON FUNZIONA IL FILTRO DELLE NEGATIVE
    for e in edges:
        source = e[0]
        target = e[1]
        label = e[2]
        my_graph.add_edge(source, target, label=label)
    return my_graph


def path_calculation(instances: List[Instance]):
    no_pair = 0
    no_path = 0
    wrong_sentences = list()
    for instance in instances:
        doc = instance.get_doc()
        html = displacy.render(doc, style='dep', page=True)
        output_path = Path('sentence.html')
        output_path.open('w', encoding='utf-8').write(html)
        root = ''
        for i in range(len(doc)):
            token = doc[i]
            ancestors = list(token.ancestors)
            if len(ancestors) == 0:
                root = token.text + '-' +str(i)
        sentences = list(doc.sents)
        depth_first_list = list()
        breadth_first_list = list()
        for s in sentences:
            s_doc = nlp(s.text)
            s_graph = graph_creation(s_doc)
            node_list = list(s_graph.nodes)
            all_edges = list(nx.edges(s_graph))
            try:
                s_depth_first = list(nx.dfs_edges(s_graph, root))
            except KeyError:
                s_depth_first = all_edges
            except NetworkXError:
                s_depth_first = all_edges
            try:
                s_breadth_first = list(nx.bfs_edges(s_graph, root))
            except KeyError:
                s_breadth_first = all_edges
            except NetworkXError:
                s_breadth_first = all_edges
            s_depth_first_list = list()
            s_breadth_first_list = list()
            for (node, next) in s_depth_first:
                edge_data = s_graph.get_edge_data(node, next)
                s_depth_first_list.append((node.rsplit('-')[0], next.rsplit('-')[0], edge_data.get('label')))
            for (node, next) in s_breadth_first:
                edge_data = s_graph.get_edge_data(node, next)
                s_breadth_first_list.append((node.rsplit('-')[0], next.rsplit('-')[0], edge_data.get('label')))
            depth_first_list = depth_first_list + s_depth_first_list
            breadth_first_list = breadth_first_list + s_breadth_first_list
        # print(depth_first_list)
        instance.set_breadth_first(breadth_first_list)
        instance.set_depth_first(depth_first_list)
        most_interesting = 0
        for i in range(len(sentences)):
            s = sentences[i]
            if 'PairDrug1' in s.text and 'PairDrug2' in s.text:
                most_interesting = i
        m_doc = nlp(sentences[most_interesting].text)
        myGraph = graph_creation(m_doc)
        # LISTA DEI LATI IN ORDINE DI COMPARIZIONE NELLA FRASE (più o meno)
        string_drug1 = ''
        string_drug2 = ''
        for i in range(len(m_doc)):
            token = m_doc[i]
            text = token.text
            # Potrei sostituire con in
            if text == 'PairDrug1':
                string_drug1 = text.lower() + '-' + str(i)
            if text == 'PairDrug2':
                string_drug2 = text.lower() + '-' + str(i)
            '''if text != 'PairDrug1' and 'PairDrug1' in text:
                print(text)
            if text != 'PairDrug2' and 'PairDrug2' in text:
                print(text)'''
        try:
            path = nx.shortest_path(myGraph, source=string_drug1, target=string_drug2)
        except NodeNotFound:
            instance.set_dependency_path(list())
            no_pair += 1
            continue
        except NetworkXNoPath:
            # Non trova il cammino dell'albero sintattico
            no_path += 1
            sentence_id = instance.get_sentence_id()
            if sentence_id not in wrong_sentences:
                wrong_sentences.append(sentence_id)
            instance.set_dependency_path(list())
            continue
        path_with_labels = list()
        for i in range(len(path)-1):
            node = path[i]
            node_split = node.rsplit('-')
            next_node = path[i+1]
            next_split = next_node.rsplit('-')
            edges = myGraph[node]
            for j in edges:
                j_split = j.rsplit('-')
                e = edges[j]
                j_label = e['label']
                if j_label == 'neg':
                    path_with_labels.append((node_split[0], j_split[0], j_label))
            edge = myGraph[node][next_node]
            edge_label = edge['label']
            path_with_labels.append((node_split[0], next_split[0], edge_label))
        #print(path_with_labels)
        instance.set_dependency_path(path_with_labels)


def negative_filtering(instances: List[Instance]) -> List[Instance]:
    path_calculation(instances)
    selected_instances = list()
    discarded = 0
    positive_discarded = 0
    for instance in instances:
        doc = instance.get_doc()
        text = doc.text
        nopair = text.count('NoPair')
        if nopair == 2:
            discarded += 1
        else:
            class_val = instance.get_class()
            dependency_path = instance.get_dependency_path()
            found = False
            for (source, target, label) in dependency_path:
                if source != 'pairdrug1' or source != 'pairdrug2' or source != 'and' or source != 'drug':
                    found = True
                if target != 'pairdrug1' or target != 'pairdrug2' or target != 'and' or target != 'drug':
                    found = True
            if not found and class_val == 'false':
                discarded += 1
            if not found and class_val != 'false':
                positive_discarded += 1
            else:
                selected_instances.append(instance)
    print("Scartate: ", discarded)
    print("Positive con albero sintattico scartabile: ", positive_discarded)
    return selected_instances


def blind_negative_filtering(instances: List[Instance]):
    path_calculation(instances)
    selected_instances = list()
    discarded_list = list()
    positive_neg = 0
    for instance in instances:
        doc = instance.get_doc()
        text = doc.text
        class_val = instance.get_class()
        nopair = text.count('NoPair')
        if nopair == 2:
            discarded_list.append(instance)
        else:
            dependency_path = instance.get_dependency_path()
            found = False
            for (source, target, label) in dependency_path:
                if source != 'pairdrug1' or source != 'pairdrug2' or source != 'and' or source != 'drug':
                    found = True
                if target != 'pairdrug1' or target != 'pairdrug2' or target != 'and' or target != 'drug':
                    found = True
            if not found:
                discarded_list.append(instance)
                if class_val != 'false':
                    positive_neg += 1
            else:
                selected_instances.append(instance)
    # print(positive_neg)
    return selected_instances, discarded_list


# 0 UNRELATED
# 1 EFFECT
# 2 MECHANISM
# 3 ADVISE
# 4 INT
def get_labelled_instances(instances) -> (List[Doc], List[int]):
    labels = []
    sents = []
    for instance in instances:
        class_val = instance.get_class()
        sent = instance.get_doc()
        sents.append(sent)
        if class_val == 'false':
            labels.append([1, 0, 0, 0, 0])
        if class_val == 'effect':
            labels.append([0, 1, 0, 0, 0])
        if class_val == 'mechanism':
            labels.append([0, 0, 1, 0, 0])
        if class_val == 'advise':
            labels.append([0, 0, 0, 1, 0])
        if class_val == 'int':
            labels.append([0, 0, 0, 0, 1])
    labels_array = np.asarray(labels, dtype='int32')
    return sents, labels_array


def to_class_int(label: str) -> int:
    if label == 'false':
        return 0
    if label == 'effect':
        return 1
    if label == 'mechanism':
        return 2
    if label == 'advise':
        return 3
    if label == 'int':
        return 4


def get_character_dictionary():
    sents = get_sentences('Dataset/Train/Overall')+get_sentences('Dataset/Test/Overall')
    characters = dict()
    index = 1
    for s in sents:
        text = s.attributes['text'].value
        for c in text:
            if c not in characters.keys():
                characters.__setitem__(c, index)
                index += 1
    # print(characters)
    import pickle
    dict_file = open('character_dict.pkl', 'wb')
    pickle.dump(characters, dict_file)
    return characters

