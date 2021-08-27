# -*- coding: utf-8 -*-

import sys, re, os, json
import unicodedata as unicode
import sklearn_crfsuite
import pickle

GRAMS_PATH = os.path.join(os.path.dirname(__file__), 'grams')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models')

BI_GRAMS = os.path.join(GRAMS_PATH, 'bi_grams.json')
TRI_GRAMS = os.path.join(GRAMS_PATH, 'tri_grams.json')
MODEL_FILE = os.path.join(MODEL_PATH,'segmentation.pkl')

def read_file(filepath):
    if not os.path.isfile(filepath):
        return None
    sentences = []
    labels = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                sent = []
                sent_labels = []
                for word in line.split():
                    syllables = word.split("_")
                    for i, syllable in enumerate(syllables):
                        sent.append(syllable)
                        if i == 0:
                            sent_labels.append('B_W')
                        else:
                            sent_labels.append('I_W')
                sentences.append(sent)
                labels.append(sent_labels)
        return {
            "sentences": sentences,
            "labels":labels
        }
    except:
        return None

def read_dir(folder):
    if not os.path.isdir(folder):
        return None
    try:
        files = os.listdir(folder)
        sentences = None
        labels = None
        for file in files:
            filepath = os.path.join(folder, file)
            if not os.path.isfiile(filepath):
                continue
            retval = read_file(filepath)
            if retval is None:
                continue
            if sentences is None:
                sentences = retval["sentences"]
                labels = retval["labels"]
            else:
                sentences += retval["sentences"]
                labels += retval["labels"]
        return {
            "sentences":sentences,
            "labels":labels
        }
    except:
        return None

def load_n_grams(filepath):
    if not os.path.isfile(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            words = json.load(f)
        if not isinstance(words, list):
            words = []
        return words
    except Exception as e:
        print("Error load_n_grams", e)
        return []

class BaseTokenizer(object):
    def tokenize(self, text):
        pass

    def get_tokenized(self, text):
        pass

    @staticmethod
    def syllablize(text):
        text = unicode.normalize('NFC', text)
        sign = ["==>", "->", "\.\.\.", ">>"]
        digits = "\d+([\.,_]\d+)+"
        email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        link = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
        datetime = [
            "\d{1,2}\/\d{1,2}(\/\d+)?",
            "\d{1,2}-\d{1,2}(-\d+)?"
        ]
        word = "\w+"
        non_word = "[^\w\s]"
        abbreviations = [
            "[A-ZĐ]+\.",
            "Tp\.",
            "Mr\.", "Mrs\.", "Ms\.",
            "Dr\.", "ThS\."
        ]
        patterns = []
        patterns.extend(abbreviations)
        patterns.extend(sign)
        patterns.extend([link, email])
        patterns.extend(datetime)
        patterns.extend([digits, non_word, word])
        patterns = "(" + "|".join(patterns) + ")"
        if sys.version_info < (3, 0):
            patterns = patterns.decode('utf-8')
        tokens = re.findall(patterns, text, re.UNICODE)
        return [token[0] for token in tokens]

class Tokenizer(BaseTokenizer):
    def __init__(self):
        self.bi_grams = load_n_grams(BI_GRAMS)
        self.tri_grams = load_n_grams(TRI_GRAMS)

    def tokenize(self, text):
        syllables = Tokenizer.syllablize(text)
        syl_len = len(syllables)
        curr_id = 0
        word_list = []
        done = False
        while (curr_id < syl_len) and (not done):
            curr_word = syllables[curr_id]
            if curr_id >= (syl_len - 1):
                word_list.append(curr_word)
                done = True
            else:
                next_word = syllables[curr_id + 1]
                pair_word = ' '.join([curr_word.lower(), next_word.lower()])
                if curr_id >= (syl_len - 2):
                    if pair_word in self.bi_grams:
                        word_list.append('_'.join([curr_word, next_word]))
                        curr_id += 2
                    else:
                        word_list.append(curr_word)
                        curr_id += 1
                else:
                    next_next_word = syllables[curr_id + 2]
                    triple_word = ' '.join([pair_word, next_next_word.lower()])
                    if triple_word in self.tri_grams:
                        word_list.append('_'.join([curr_word, next_word, next_next_word]))
                        curr_id += 3
                    elif pair_word in self.bi_grams:
                        word_list.append('_'.join([curr_word, next_word]))
                        curr_id += 2
                    else:
                        word_list.append(curr_word)
                        curr_id += 1
        return word_list

class CrfTokenizer(BaseTokenizer):
    def __init__(self):
        self.bi_grams = load_n_grams(BI_GRAMS)
        self.tri_grams = load_n_grams(TRI_GRAMS)
        self.features_cfg_arr = [
            ['word.tri_gram'],
            ['word.lower', 'word.isupper', 'word.istitle', 'word.bi_gram'],
            ['bias', 'word.lower', 'word.isupper', 'word.istitle', 'word.isdigit'],
            ['word.lower', 'word.isupper', 'word.istitle', 'word.bi_gram'],
            ['word.tri_gram']
        ]
        self.center_id = int((len(self.features_cfg_arr) - 1) / 2)
        self.function_dict = {
            'bias': lambda word, *args: 1.0,
            'word.lower': lambda word, *args: word.lower(),
            'word.isupper': lambda word, *args: word.isupper(),
            'word.istitle': lambda word, *args: word.istitle(),
            'word.isdigit': lambda word, *args: word.isdigit(),
            'word.bi_gram': lambda word, word1, relative_id, *args: self._check_bi_gram([word, word1], relative_id),
            'word.tri_gram': lambda word, word1, word2, relative_id, *args: self._check_tri_gram([word, word1, word2], relative_id)
        }

    def _check_bi_gram(self, a, relative_id):
        if relative_id < 0:
            return ' '.join([a[0], a[1]]).lower() in self.bi_grams
        return ' '.join([a[1], a[0]]).lower() in self.bi_grams

    def _check_tri_gram(self, b, relative_id):
        if relative_id < 0:
            return ' '.join([b[0], b[1], b[2]]).lower() in self.tri_grams
        return ' '.join([b[2], b[1], b[0]]).lower() in self.tri_grams

    def _get_base_features(self, features_cfg_arr, word_list, relative_id=0):
        prefix = ""
        if relative_id < 0:
            prefix = str(relative_id) + ":"
        elif relative_id > 0:
            prefix = '+' + str(relative_id) + ":"
        features_dict = dict()
        for ft_cfg in features_cfg_arr:
            features_dict.update({prefix+ft_cfg: wrapper(self.function_dict[ft_cfg], word_list + [relative_id])})
        return features_dict

    @staticmethod
    def _check_special_case(word_list):
        if word_list[1].istitle() and (not word_list[0].istitle()):
            return True
        for word in word_list:
            if word in string.punctuation:
                return True

        for word in word_list:
            if word[0].isdigit():
                return True

        return false

    def create_syllable_features(self, text, word_id):
        word = text[word_id]
        features_dict = self._get_base_features(self.features_cfg_arr[self.center_id], [word])
        if word_id > 0:
            word1 = text[word_id - 1]
            features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id - 1],[word1, word], -1))

            if word_id > 1:
                word2 = text[word_id - 2]
                features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id - 2],[word2, word1, word], -2))

        if word_id < len(text) - 1:
            word1 = text[word_id + 1]
            features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id + 1],[word1, word], +1))

            if word_id < len(text) - 2:
                word2 = text[word_id + 2]
                features_dict.update(self._get_base_features(self.features_cfg_arr[self.center_id + 2],[word2, word1, word], +2))

        return features_dict

    def create_sentence_features(self, prepared_sentence):
        return [self.create_syllable_features(prepared_sentence, i) for i in range(len(prepared_sentence))]

    def prepare_training_data(self, data):
        X, y = [], []
        for i, sent in enumerate(data["sentences"]):
            X.append(self.create_sentence_features(sent))
            y.append(data["labels"][i])
        return X, y

    def train(self, data):
        if os.path.isdir(data):
            retval = load_dir(data)
        else:
            retval = load_file(data)

        if retval is None:
            return None

        X, y = self.prepare_training_data(retval)
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        crf.fit(X, y)
        try:
            with open(MODEL_FILE,'wb') as f:
                pickle.dump(crf, f)
        except:
            pass

    def load_tagger(self):
        try:
            with open(MODEL_FILE,'wb') as f:
                self.tagger = pickle.load(f)
        except:
            pass

    def tokenize(self, text):
        if self.tagger is None:
            self.load_tagger()
        sent = self.syllablize(text)
        syl_len = len(sent)
        if syl_len <= 1:
            return sent

        if self.tagger is None:
            return sent

        test_features = self.create_sentence_features(sent)
        prediction = self.tagger.predict([test_features])[0]

        word_list = []
        pre_word = sent[0]
        for i, p in enumerate(prediction[1:], start=1):
            if p == 'I_W' and not self._check_special_case(sent[i-1:i+1]):
                pre_word += "_" + sent[i]
                if i == (syl_len - 1):
                    word_list.append(pre_word)
            else:
                if i > 0:
                    word_list.append(pre_word)
                if i == (syl_len - 1):
                    word_list.append(sent[i])
                pre_word = sent[i]
        return word_list

    def get_tokenized(self, text):
        if self.tagger is None:
            self.load_tagger()
        if self.tagger is None:
            return text
        sent = self.syllablize(text)
        if len(sent) <= 1:
            return text
        test_features = self.create_sentence_features(sent)
        prediction = self.tagger.predict([test_features])[0]
        complete = sent[0]
        for i, p in enumerate(prediction[1:], start=1):
            if p == 'I_W' and not self._check_special_case(sent[i-1:i+1]):
                complete = complete + '_' + sent[i]
            else:
                complete = complete + ' ' + sent[i]
        return complete

def test_tokenizer(text):
    tokenizer = Tokenizer()
    tokens = lm_tokenizer.tokenize(text)
    print(tokens)
