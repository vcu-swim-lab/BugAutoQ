import observed_behavior_rule as ob
import spacy
from spacy.matcher import Matcher
from difflib import Differ


class Calculator(object):

    def __init__(self, threshold=0.0001):
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab, validate=True)
        self.threshold = threshold
        ob.setup_s_ob_neg_aux_verb(self.matcher)
        ob.setup_s_ob_verb_error(self.matcher)
        ob.setup_s_ob_neg_verb(self.matcher)
        ob.setup_s_ob_but(self.matcher)
        ob.setup_s_ob_cond_pos(self.matcher)

    def utility(self, answer, post):
        answer_ob = self._get_ob(answer)
        if len(answer_ob) == 0:
            return self.threshold

        post_ob = self._get_ob(post)
        diff = self._get_diff(post_ob.splitlines(keepends=True), answer_ob.splitlines(keepends=True))
        # it doesnt make sense to put softmax on one number
        util = len(diff.split()) / float(len(answer_ob.split()))
        return max(util, self.threshold)

    def _get_diff(self, text_a, text_b):
        differ = Differ()
        diff_lines = differ.compare(text_a, text_b)
        diff = ' '.join([diff for diff in diff_lines if diff.startswith('+ ')]).replace('+ ', ' ')
        return diff

    def _get_ob(self, text):
        ob_str = ''
        for sentence in self.nlp(text).sents:
            sent = sentence.text.strip()
            if sent.startswith('>') or sent.endswith('?'):
                continue
            else:
                sent_nlp = self.nlp(sent)
                matches = self.matcher(sent_nlp)
                if len(matches) >= 1:
                    ob_str += sent + '\n'
        return ob_str
