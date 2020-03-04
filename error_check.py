from relation_format_extraction import sentences_from_prediction
from pre_processing_lib import get_sentences


def has_numbers(string):
    return any(char.isdigit() for char in string)


def get_entity_dict():
    xml_sentences = get_sentences('Dataset/Test/Overall')
    entity_dict = dict()
    for i in range(0, len(xml_sentences)):
        entities = xml_sentences[i].getElementsByTagName('entity')
        for entity in entities:
            id = entity.attributes['id'].value
            text = entity.attributes['text'].value
            entity_dict.__setitem__(id, text)
    return entity_dict


sentences = sentences_from_prediction('inputSent2.txt', 'predLabels2_modified.txt')
entity_dict = get_entity_dict()
only_capital = []
dash = []
numbers = []
parenthesis = []
for id in entity_dict.keys():
    text : str = entity_dict.get(id)
    if text.upper() == text:
        only_capital.append(text)
    if '-' in text:
        dash.append(text)
    if has_numbers(text):
        numbers.append(text)
    if '(' in text and ')' in text:
        parenthesis.append(text)
total_approximate = []
total_missing = []
for s in sentences:
    total_approximate += s.approximate_drugs
    total_missing += s.missing_drugs
text_approximate = []
text_missing = []
for id in total_approximate:
    text_approximate.append(entity_dict.get(id))
for id in total_missing:
    text_missing.append(entity_dict.get(id))
num = 0
for d in dash:
    if d in text_missing or d in text_approximate:
       num += 1
print(num / len(dash))
num = 0
for n in numbers:
    if n in text_approximate or n in text_missing:
        num += 1
print(num/len(numbers))
num = 0
for o in only_capital:
    if o in text_approximate or o in text_missing:
        num += 1
print(num/len(only_capital))
num = 0
for p in parenthesis:
    if p in text_approximate or p in text_missing:
        num += 1
print(num/len(parenthesis))
print(text_missing + text_approximate)