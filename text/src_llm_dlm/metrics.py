"""
Metrics.
"""
import re
from collections import defaultdict

from nltk import ngrams
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

from src_llm_dlm.utils import remove_parenthesis

_LEMMA_TOKEN_RE = re.compile(r"[a-z']+")


def score_fmt_adh(text: str) -> float:
    # [^=\n]* was changed to [^\n]* on 2026-07-14 — the old pattern rejected
    # any '=' inside the body (e.g. "πλάξ = plate"), causing false fmt_adh=0.
    FMT_RE = r'==\s*[^=\s\n][^\n]*=='
    return 1.0 if re.search(FMT_RE, text or "") else 0.0


def _word_boundary_match(word, text):
    # Word-boundary matching instead of plain substring: avoids false hits like
    # "car" inside "carriage"/"cart" or "alcohol" inside "non-alcoholic", while
    # still catching whole-word hyponym matches (e.g. "boat" in "a dinghy is a
    # small boat"). Known gap: fused compounds with no internal boundary (e.g.
    # "boat" inside "sailboat") won't match -- see findings/wordnet_checker_failure_modes.md.
    return re.search(r'\b' + re.escape(word) + r'\b', text) is not None


def _text_noun_lemmas(text):
    # WordNet's own morphy noun-lemmatizer, used alongside (not instead of) naive
    # '+s' pluralization: morphy fixes irregular plurals like "potatoes" -> "potato",
    # but is inconsistent (e.g. "chips" stays "chips", not "chip", since WordNet also
    # lists "chips" as its own lemma) so '+s' is kept as a second, independent check.
    # Restricting to noun morphology (pos='n') matters: it's what keeps "stewed"
    # (a verb form) from collapsing back to "stew" and reintroducing that false positive.
    return {wordnet.morphy(t, wordnet.NOUN) or t for t in _LEMMA_TOKEN_RE.findall(text)}


def _word_matches(word, text, lemmas):
    # Single-word terms rely on the noun-lemma set (so morphy-normalized forms count)
    # in addition to the naive '+s' plural; multi-word terms have no lemma entry, so
    # they're checked as whole-phrase word-boundary matches instead.
    if ' ' not in word:
        return word in lemmas or _word_boundary_match(word + 's', text)
    return _word_boundary_match(word, text) or _word_boundary_match(word + 's', text)


def _match_words(words, text, lemmas):
    return any(_word_matches(w, text, lemmas) for w in words)


def _wordnet_constraint_check(generated_text, topic, constraint, hierarchy):
    generated_text = generated_text.lower()
    lemmas = _text_noun_lemmas(generated_text)

    forbidden_words = set(hierarchy[constraint] + [constraint])
    topical_words = set(hierarchy[topic])

    violated = _match_words(forbidden_words, generated_text, lemmas)
    on_topic = _match_words(topical_words, generated_text, lemmas)

    return generated_text, violated, on_topic, None


def _wikidata_constraint_check(generated_text, topic, constraint, hierarchy):
    forbidden_words = [remove_parenthesis(q_title) for _, q_title in hierarchy[constraint]]
    violated, violated_word = False, None
    for w in forbidden_words:
        if w in generated_text:
            violated, violated_word = True, w
            break

    topical_words = [remove_parenthesis(q_title) for _, q_title in hierarchy[topic]]
    on_topic, on_topic_word = False, None
    for w in topical_words:
        if w in generated_text:
            on_topic, on_topic_word = True, w
            break

    return generated_text, violated, on_topic, dict(violated_word=violated_word, on_topic_word=on_topic_word)


def compute_prediction_metrics(prediction, hierarchy, data_mode):
    datapoint = prediction['datapoint']
    context = datapoint['context']
    topic = datapoint['topic']
    constraint = datapoint['constraint']
    generated_text = prediction['generated_text']

    if data_mode == 'wordnet':
        generated_text, violated, on_topic, extracted = _wordnet_constraint_check(
            generated_text, topic, constraint, hierarchy
        )
    elif data_mode == 'wikidata':
        generated_text, violated, on_topic, extracted = _wikidata_constraint_check(
            generated_text, topic, constraint, hierarchy
        )
    else:
        raise ValueError(f'Data mode {data_mode} not recognized.')

    prediction_metric = dict(
        violated=violated,
        on_topic=on_topic,
        on_topic_not_violated=(not violated) and on_topic,
        copying_bleu_score=copying_bleu(context, generated_text),
        exact_copy=exact_copy(context, generated_text),
        fmt_adh=score_fmt_adh(generated_text),
        gen_length=len(generated_text.split()),
    )
    prediction_metric.update(get_repetition_scores(generated_text.split()))

    return dict(
        prediction_metric=prediction_metric,
        extracted=extracted,
    )


def copying_bleu(context, generated_text):
    prediction = word_tokenize(generated_text.strip('==').strip())
    max_score = float('-inf')
    for context_sent in context.strip().split('\n'):
        references = [word_tokenize(context_sent.strip('==').strip())]
        score = sentence_bleu(references, prediction)
        if score > max_score:
            max_score = score
    return max_score


def exact_copy(context, generated_text):
    gen_norm = ' '.join(generated_text.strip('=').strip().lower().split())
    for line in context.strip().split('\n'):
        if ' '.join(line.strip('=').strip().lower().split()) == gen_norm:
            return True
    return False


def get_repetition_scores(tokens):
    metric = defaultdict(float)
    for n in range(1, 5):
        ngs = list(ngrams(tokens, n))
        unique_ngs = set(ngs)
        if not ngs:
            metric[f'seq-rep-{n}'] = 0.0
        else:
            metric[f'seq-rep-{n}'] = 1.0 - (len(unique_ngs) / len(ngs))
    return dict(metric)
