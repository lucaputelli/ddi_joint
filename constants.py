from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
import spacy
from spacy.language import Language, Tokenizer


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