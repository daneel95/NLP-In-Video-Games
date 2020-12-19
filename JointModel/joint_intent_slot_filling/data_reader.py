import joint_intent_slot_filling.constants as consts
import pandas as pd
import numpy as np
import string

from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

DATA_COLUMN = 'text'
INTENT_COLUMN = 'intent'
SLOT_COLUMN = 'slots'


class JointIntentSlotFillingData:
    def __init__(self, tokenizer: FullTokenizer,
                 training_data_path=consts.TRAINING_DATA_PATH,
                 test_data_path=consts.TEST_DATA_PATH):
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.__read_data()
        self.tokenizer = tokenizer
        self.max_sequence_length = 0

        # Tokenize input data
        self.train_input_ids, self.train_input_masks, self.train_segment_ids, self.train_valid_positions = \
            self.__tokenize_input_data(self.training_input_data)
        self.test_input_ids, self.test_input_masks, self.test_segment_ids, self.test_valid_positions = \
            self.__tokenize_input_data(self.test_input_data)

        # Pad input data
        self.train_input_ids, self.train_input_masks, self.train_segment_ids, self.train_valid_positions = \
            self.__padding_input_data(
                self.train_input_ids,
                self.train_input_masks,
                self.train_segment_ids,
                self.train_valid_positions)
        self.test_input_ids, self.test_input_masks, self.test_segment_ids, self.test_valid_positions = \
            self.__padding_input_data(self.test_input_ids,
                                      self.test_input_masks,
                                      self.test_segment_ids,
                                      self.test_valid_positions)

        # Slots tags encoder
        self.__train_slots_tags_encoder(self.training_slots_data, self.test_slots_data)
        self.train_slots_tokens = self.__tokenize_slots_data(self.training_slots_data, self.train_valid_positions)
        self.test_slots_tokens = self.__tokenize_slots_data(self.test_slots_data, self.test_valid_positions)

        # Intent encoder
        self.__train_intent_encoder(self.training_intent_data, self.test_intent_data)
        self.train_intent_tokens = self.__tokenize_intent_data(self.training_intent_data)
        self.test_intent_tokens = self.__tokenize_intent_data(self.test_intent_data)

    # just reads data as pandas dataframe from paths
    def __read_data(self):
        training_data = pd.read_csv(self.training_data_path)
        test_data = pd.read_csv(self.test_data_path)

        # Training data
        self.training_input_data = training_data[DATA_COLUMN]
        self.training_intent_data = training_data[INTENT_COLUMN]
        self.training_slots_data = training_data[SLOT_COLUMN]

        # Test data
        self.test_input_data = test_data[DATA_COLUMN]
        self.test_intent_data = test_data[INTENT_COLUMN]
        self.test_slots_data = test_data[SLOT_COLUMN]

    def __train_slots_tags_encoder(self, train_slots, test_slots):
        self.slots_label_encoder = LabelEncoder()
        data = ['[padding]', '[CLS]', '[SEP]'] + \
               [item for sublist in [s.split() for s in train_slots] for item in sublist] + \
               [item for sublist in [s.split() for s in test_slots] for item in sublist]

        self.slots_label_encoder.fit(data)

    def __train_intent_encoder(self, train_intents, test_intents):
        self.intent_label_encoder = LabelEncoder()
        self.intent_label_encoder.fit(train_intents.tolist() + test_intents.tolist())

    def __tokenize_input_data(self, data):
        input_ids = []
        input_masks = []
        valid_positions = []
        segment_ids = []
        for text in data:
            # Be agnostic of punctuation, for now
            # TODO: Check how to do it with punctuation -- For now leave without punctuation
            text = text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            self.max_sequence_length = max(self.max_sequence_length, len(token_ids))

            input_ids.append(token_ids)
            input_masks.append([1] * (len(token_ids)))
            valid_positions.append([0 if "##" in el else 1 for el in tokens])
            segment_ids.append([0] * len(tokens))

        return np.array(input_ids), np.array(input_masks), np.array(segment_ids), np.array(valid_positions)

    def __padding_input_data(self, ids, masks, segments, valid_positions):
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(ids,
                                                                  maxlen=self.max_sequence_length,
                                                                  truncating='post',
                                                                  padding='post')
        input_masks = tf.keras.preprocessing.sequence.pad_sequences(masks,
                                                                    maxlen=self.max_sequence_length,
                                                                    truncating='post',
                                                                    padding='post')
        input_segments = tf.keras.preprocessing.sequence.pad_sequences(segments,
                                                                       maxlen=self.max_sequence_length,
                                                                       truncating='post',
                                                                       padding='post')
        input_valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions,
                                                                              maxlen=self.max_sequence_length,
                                                                              truncating='post',
                                                                              padding='post')

        return input_ids, input_masks, input_segments, input_valid_positions

    def __tokenize_slots_data(self, data, valid_positions):
        split_data = [s.split() for s in data]
        labels = [self.slots_label_encoder.transform(['[CLS]'] + x + ['[SEP]']).astype(np.int32) for x in split_data]

        output = []
        for j, (label, valid_position) in enumerate(zip(labels, valid_positions)):
            index = 0
            label_output = []
            for i in range(len(valid_position)):
                if valid_position[i] == 1:
                    label_output.append(label[index])
                    index += 1
                else:
                    label_output.append(0)
            output.append(label_output)

        return np.array(output)

    def __tokenize_intent_data(self, data):
        return self.intent_label_encoder.transform(data).astype(np.int32)
