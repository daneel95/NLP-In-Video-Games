import numpy as np
from bert.tokenization.bert_tokenization import FullTokenizer
from tqdm import tqdm


class IntentClassificationData:
    DATA_COLUMN = 'text'
    LABEL_COLUMN = 'intent'

    def __init__(self,
                 train,
                 test,
                 tokenizer: FullTokenizer,
                 classes,
                 max_sequence_length=192):
        self.tokenizer = tokenizer
        self.max_sequence_length = 0
        self.classes = classes

        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        self.max_sequence_length = min(self.max_sequence_length, max_sequence_length)
        self.train_x, self.test_x = map(self._padding, [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        for _, row in tqdm(df.iterrows()):
            text, label = row[IntentClassificationData.DATA_COLUMN], row[IntentClassificationData.LABEL_COLUMN]

            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            self.max_sequence_length = max(self.max_sequence_length, len(token_ids))

            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _padding(self, ids):
        x = []

        for input_ids in ids:
            cut_point = min(len(input_ids), self.max_sequence_length - 2)
            input_ids = input_ids[:cut_point]
            input_ids = input_ids + [0] * (self.max_sequence_length - len(input_ids))
            x.append(np.array(input_ids))

        return np.array(x)
