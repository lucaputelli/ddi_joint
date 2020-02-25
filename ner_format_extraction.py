from pre_processing_lib import get_sentences, number_substitution
import spacy
from spacy.language import Language, Tokenizer
from typing import Dict


prefix_re = spacy.util.compile_prefix_regex(Language.Defaults.prefixes + (';', '\*', '\(', '\)'))
suffix_re = spacy.util.compile_suffix_regex(Language.Defaults.suffixes + (';' , '\*', '\(', '\)'))
infix_re = spacy.util.compile_infix_regex(Language.Defaults.infixes + ("/", "-", ";", "\*", '\(', '\)'))
nlp = spacy.load('en')
nlp.tokenizer.suffix_search = suffix_re.search
nlp.tokenizer.prefix_search = prefix_re.search
nlp.tokenizer.infix_finditer = infix_re.finditer


class SequencePair:

    def __init__(self, first_sequence, second_sequence):
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence


def only_spaces(token: str) -> bool:
    for i in range(len(token)):
        if token[i] != ' ':
            return False
    return True


def ner_format(sentences, no_number = False, clean_separators = False) -> Dict:
    ner_instances = dict()
    for i in range(len(sentences)):
        s_id = sentences[i].attributes['id'].value
        entities = sentences[i].getElementsByTagName('entity')
        entity_tuples = list()
        text = str(sentences[i].attributes['text'].value)
        # NUMBER SUBSTITUTION
        if no_number:
            text = number_substitution(text)
        nlp_doc = nlp(text)
        tokens = list(nlp_doc)
        cleaned_tokens = [t for t in tokens if '\n' not in t.text or '\r' not in t.text]
        tokens = cleaned_tokens
        for token in tokens:
            if ';' in token.text and token.text != ';':
                print(token.text)
        # displacy.serve(nlp_doc, style='dep')
        for j in range(len(entities)):
            e_id = entities[j].attributes['id'].value
            e_text = entities[j].attributes['text'].value
            offset_string = entities[j].attributes['charOffset'].value
            type = entities[j].attributes['type'].value
            split = str.split(offset_string, ';')
            if len(split) > 1:
                first = int(split[0].split('-')[0])
                last = int(split[1].split('-')[1])
                entity = (e_text, first, last, type, e_id)
                entity_tuples += [entity]
                continue
            for s in split:
                offsets = str.split(s, "-")
                left = int(offsets.__getitem__(0))
                right = int(offsets.__getitem__(1))
                entity = (e_text, left, right, type, e_id)
                entity_tuples += [entity]
        drug_dict = dict()
        for entity in entity_tuples:
            left_tuple = entity[1]
            right_tuple = entity[2]
            type = entity[3]
            id = entity[4]
            for k in range(len(tokens)):
                try:
                    t = tokens.__getitem__(k)
                except IndexError:
                    print(tokens)
                    continue
                left_idx = t.idx
                length = len(t.text)
                right_idx = t.idx + length - 1
                if left_tuple == left_idx:
                    if right_idx == right_tuple:
                        drug_dict.__setitem__(k, (0, type, id))
                        break
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
                            # span = nlp_doc[k: k + n]
                            # spans_merge.append(span)
                            drug_dict.__setitem__(k, (n, type, id))
                            break
        ner_index = dict()
        for i in range(len(tokens)):
            if i in drug_dict.keys():
                (span_index, type, id) = drug_dict.get(i)
                if span_index > 0:
                    for j in range(span_index):
                        ner_index.__setitem__(i+j, (type, id))
                else:
                    ner_index.__setitem__(i, (type, id))
            else:
                if i not in ner_index.keys():
                    ner_index.__setitem__(i, ('O', 'O'))
        ner_format = [(tokens[i], ner_index.get(i)) for i in ner_index.keys() if not only_spaces(tokens[i].text)]
        # RIMUOVO SEPARATORI DAI NOMI
        if clean_separators:
            separators = ['(', ')', '-']
            cleaner_format = []
            for token, label_tuple in ner_format:
                if label_tuple != ('O', 'O') and token.text in separators:
                    continue
                else:
                    cleaner_format.append((token, label_tuple))
            ner_instances.__setitem__(s_id, cleaner_format)
        else:
            ner_instances.__setitem__(s_id, ner_format)
    return ner_instances


def write_ner_dataset(sents, file_name, no_number=False, clean_separators=False):
    f = open(file_name+'.txt', 'w')
    csv_file = open('csv_'+file_name+'.csv', 'w')
    other_file = open('other_file.txt', 'w')
    b_i = 0
    ner_format_dictionary = ner_format(sents, no_number, clean_separators)
    for sentence_key in ner_format_dictionary:
        instance = ner_format_dictionary.get(sentence_key)
        csv_file.write(sentence_key+';;\n')
        string = sentence_key+': ['
        for t in instance:
            key, (type, id) = t
            key_text = key.text
            if key_text == ';':
                key_text = ','
            if type == 'O':
                b_i = 0
                new_type = type
            else:
                if b_i == 0:
                    new_type = 'B-' + type
                    b_i = 1
                else:
                    new_type = 'I-' + type
            # print(key.text + '\t' + new_type)
            f.write(key.text + '\t' + new_type + '\n')
            id_label = new_type
            if new_type.startswith('B'):
                id_label = 'B-'+id
            elif new_type.startswith('I'):
                id_label = 'I-'+id
            if instance.index(t) != len(instance)-1:
                string += id_label + ', '
            else:
                string += id_label
            if new_type.startswith('B'):
                csv_file.write(key_text+';'+new_type+';B-'+id+'\n')
            elif new_type.startswith('I'):
                csv_file.write(key_text + ';' + new_type + ';I-' + id + '\n')
            else:
                csv_file.write(key_text + ';' + new_type + ';' + new_type + '\n')
        f.write('\n')
        string += ']\n'
        other_file.write(string)
    f.close()


def double_sequence(sentences):
    sequence_pairs = list()
    intersected = list()
    for i in range(len(sentences)):
        entities = sentences[i].getElementsByTagName('entity')
        pairs = sentences[i].getElementsByTagName('pair')
        entity_tuples = list()
        text = str(sentences[i].attributes['text'].value)
        nlp_doc = nlp(text)
        tokens = list(nlp_doc)
        for j in range(len(entities)):
            e_id = entities[j].attributes['id'].value
            e_text = entities[j].attributes['text'].value
            offset_string = entities[j].attributes['charOffset'].value
            type = entities[j].attributes['type'].value
            split = str.split(offset_string, ';')
            if len(split) > 1:
                intersected.append(e_id)
            else:
                for s in split:
                    offsets = str.split(s, "-")
                    left = int(offsets.__getitem__(0))
                    right = int(offsets.__getitem__(1))
                    entity = (e_text, left, right, type, e_id)
                    entity_tuples += [entity]
        drug_dict = dict()
        for entity in entity_tuples:
            left_tuple = entity[1]
            right_tuple = entity[2]
            type = entity[3]
            id = entity[4]
            found = False
            for k in range(len(tokens)):
                try:
                    t = tokens.__getitem__(k)
                except IndexError:
                    print(tokens)
                    continue
                left_idx = t.idx
                length = len(t.text)
                right_idx = t.idx + length - 1
                if left_tuple == left_idx:
                    if right_idx == right_tuple:
                        drug_dict.__setitem__(k, (0, type, id))
                        found = True
                        # print(t)
                    else:
                        n = 1
                        # print(right_tuple, right_idx)
                        while right_idx <= right_tuple:
                            if k + n >= len(tokens):
                                break
                            next = tokens.__getitem__(k + n)
                            right_idx = next.idx + len(next.text)
                            n = n + 1
                        if right_idx >= right_tuple:
                            span = nlp_doc[k: k + n]
                            print(entity[0])
                            print(span)
                            drug_dict.__setitem__(k, (n, type, id))
                            found = True
            if not found:
                print(entity[0])
        id_dictionary = dict()
        for k in drug_dict.keys():
            token = nlp_doc[k]
            (n, type, id) = drug_dict.get(k)
            id_dictionary.__setitem__(id, (token, k, n, type))
        pairs_dict = dict()
        for p in pairs:
            id = p.attributes['id'].value
            e1 = p.attributes['e1'].value
            e2 = p.attributes['e2'].value
            ddi = p.attributes['ddi'].value
            if ddi == 'true':
                try:
                    type = p.attributes['type'].value
                except KeyError:
                    type = 'int'
            else:
                type = 'unrelated'
            if e1 not in pairs_dict.keys():
                pairs_dict.__setitem__(e1, [(e2, type)])
            else:
                pairs_dict.get(e1).append((e2, type))
        first_sequence = dict()
        second_sequences = list()
        for e1 in pairs_dict.keys():
            e2_list = [tuple[0] for tuple in pairs_dict.get(e1)]
            type_list = [tuple[1] for tuple in pairs_dict.get(e1)]
            second_sequence = dict()
            for j in range(len(e2_list)):
                e2 = e2_list[j]
                rel_type = type_list[j]
                if e2 in intersected:
                    continue
                if e2 not in id_dictionary.keys():
                    print(e2)
                    continue
                (token, k, n, drug_type) = id_dictionary.get(e2)
                for i in range(len(nlp_doc)):
                    if i == k:
                        span_index = n
                        if span_index > 0:
                            for j in range(span_index):
                                second_sequence.__setitem__(i + j, rel_type)
                        else:
                            second_sequence.__setitem__(i, rel_type)
            for i in range(len(nlp_doc)):
                if i not in second_sequence.keys():
                    second_sequence.__setitem__(i, 'O')
            second_sequence = [(nlp_doc[i], second_sequence.get(i)) for i in range(len(nlp_doc))]
            second_sequences.append(second_sequence)
            if e1 in intersected or e1 not in id_dictionary:
                continue
            (token, k, span_index, drug_type) = id_dictionary.get(e1)
            for i in range(len(nlp_doc)):
                if i == k:
                    if span_index > 0:
                        for j in range(span_index):
                            if j == 0:
                                first_sequence.__setitem__(i + j, 'B-'+drug_type)
                            else:
                                first_sequence.__setitem__(i + j, 'I-' + drug_type)
                    else:
                        first_sequence.__setitem__(i, 'B-'+drug_type)
        for i in range(len(nlp_doc)):
            if i not in first_sequence.keys():
                first_sequence.__setitem__(i, 'O')
        first_sequence = [(nlp_doc[i], first_sequence.get(i)) for i in range(len(nlp_doc))]
        for second_sequence in second_sequences:
            instance = SequencePair(first_sequence, second_sequence)
            sequence_pairs.append(instance)
    return sequence_pairs


def iob2_format():
    csv_file = open('MANUALLY_CHECKED_CSV_TEST.csv', 'r')
    iob2_file = open('MANUALLY_CHECKED_TEST_SET_IOB2.txt', 'w')
    lines = csv_file.readlines()
    for i in range(1, len(lines)):
        if lines[i].startswith('DDI-'):
            iob2_file.write('\n')
        else:
            split = lines[i].split(';')
            token = split[0]
            label = split[1]
            iob2_file.write(token + '\t' + label + '\n')
    iob2_file.close()


def cleaned_iob2():
    csv_file = open('MANUALLY_CHECKED_CSV_TEST.csv', 'r')
    separators = ['(', ')', '-']
    lines = csv_file.readlines()
    iob2_tuples = []
    lines[0] = lines[0].replace('\n', '')
    for i in range(1, len(lines)):
        lines[i] = lines[i].replace('\n', '')
        if lines[i].startswith('DDI-'):
            iob2_tuples.append(('\n', '\n'))
        else:
            split = lines[i].split(';')
            token = split[0]
            label = split[1]
            id = split[2]
            if token in separators and label != 'O':
                continue
            else:
                iob2_tuples.append((token, label))
    for i in range(1, len(iob2_tuples)):
        token, label = iob2_tuples[i-1]
        next_token, next_label = iob2_tuples[i]
        if next_label.startswith('I') and label == 'O':
            iob2_tuples[i] = (next_token, next_label.replace('I', 'B'))
    iob2_file = open('CLEANED_CHECKED_TEST_SET_IOB2.txt', 'w')
    for token, label in iob2_tuples:
        if token == '\n':
            iob2_file.write('\n')
        else:
            iob2_file.write(token + '\t' + label + '\n')
    iob2_file.close()


def cleaned_linear():
    separators = ['(', ')', '-']
    csv_file = open('MANUALLY_CHECKED_CSV_TEST.csv', 'r')
    lines = csv_file.readlines()
    token_line = ''
    id_line = ''
    label_line = ''
    token_file = open('CLEANED_CHECKED_TOKEN.txt', 'w')
    id_file = open('CLEANED_CHECKED_ID.txt', 'w')
    label_file = open('CLEANED_CHECKED_LABEL.txt', 'w')
    for i in range(0, len(lines)-1):
        lines[i] = lines[i].replace('\n', '')
        if lines[i].startswith('DDI-'):
            if i != 0:
                token_file.write(token_line+'\n')
                label_file.write(label_line+'\n')
                id_file.write(id_line+'\n')
            token_line = lines[i].replace(';', '')
            token_line += ': ['
            id_line = token_line
            label_line = token_line
        else:
            split = lines[i].split(';')
            token = number_substitution(split[0])
            label = split[1]
            id = split[2]
            previous_split = lines[i-1].split(';')
            # previous_tokens = previous_split[0]
            previous_label = previous_split[1]
            # previous_id = previous_split[2]
            if token in separators and label != 'O':
                continue
            if previous_label == 'O' and label.startswith('I'):
                label = label.replace('I', 'B')
                id = id.replace('I', 'B')
            if lines[i+1].startswith('DDI-'):
                token_line += token + ']'
                id_line += id + ']'
                label_line += label + ']'
            else:
                token_line += token + ', '
                id_line += id + ', '
                label_line += label + ', '
    token_file.close()
    id_file.close()
    label_file.close()


def linear_format():
    csv_file = open('csv_TRAINING_SET.csv', 'r')
    lines = csv_file.readlines()
    token_line = ''
    id_line = ''
    label_line = ''
    token_file = open('TRAINING_TOKEN.txt', 'w')
    id_file = open('TRAINING_ID.txt', 'w')
    label_file = open('TRAINING_LABEL.txt', 'w')
    for i in range(0, len(lines)-1):
        lines[i] = lines[i].replace('\n', '')
        if lines[i] == '':
            continue
        if lines[i].startswith('DDI-'):
            if i != 0:
                token_file.write(token_line+'\n')
                label_file.write(label_line+'\n')
                id_file.write(id_line+'\n')
            token_line = lines[i].replace(';', '')
            token_line += ': ['
            id_line = token_line
            label_line = token_line
        else:
            split = lines[i].split(';')
            token = split[0]
            label = split[1]
            id = split[2]
            if lines[i+1].startswith('DDI-'):
                token_line += token + ']'
                id_line += id + ']'
                label_line += label + ']'
            else:
                token_line += token + ', '
                id_line += id + ', '
                label_line += label + ', '
    token_file.close()
    id_file.close()
    label_file.close()


# sentences = get_sentences('Dataset/Train/Overall')
# write_ner_dataset(sentences, 'TRAINING_SET')
# linear_format()
# sequence_pairs = double_sequence(sentences)
# print(sequence_pairs)
# ner_instances = ner_format(sentences, True, True)
# write_ner_dataset(sentences, 'train', True, True)
# cleaned_iob2()
# cleaned_linear()