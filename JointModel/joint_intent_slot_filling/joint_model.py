import os
import string
from itertools import chain

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joint_intent_slot_filling.data_reader import JointIntentSlotFillingData
import joint_intent_slot_filling.constants as consts
import joint_intent_slot_filling.download_models as dm

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dropout, Dense, TimeDistributed, Multiply, Layer
from tensorflow.keras.models import Model
import tensorflow_hub as hub

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from bert.tokenization.bert_tokenization import FullTokenizer

# Set the tensorflow hub cache directory
os.environ['TFHUB_CACHE_DIR'] = 'joint_intent_slot_filling/tf_cache'


class JointIntentClassificationSlotsFillingModel:
    def __init__(self, is_training=False,
                 training_data_path=consts.TRAINING_DATA_PATH,
                 test_data_path=consts.TEST_DATA_PATH,
                 trained_model_directory=consts.TRAINED_MODEL_DIRECTORY,
                 trained_model_max_sequence_file=consts.TRAINED_MODEL_MAX_SEQUENCE_FILE,
                 slots_label_encoder_file=consts.SLOTS_LABEL_ENCODER_FILE,
                 intent_label_encoder_file=consts.INTENT_LABEL_ENCODER_FILE,
                 model_metrics_directory=consts.MODEL_METRICS_DIRECTORY,
                 model_metrics_report_file=consts.MODEL_METRICS_REPORT_FILE,
                 model_confusion_matrix_file=consts.MODEL_CONFUSION_MATRIX_FILE,
                 is_old=False):
        # Setup custom train and test data paths and other paths. Default are the new ones
        self.training_data_path = training_data_path
        self.test_data_path = test_data_path
        self.trained_model_directory = trained_model_directory
        self.trained_model_max_sequence_file = trained_model_max_sequence_file
        self.slots_label_encoder_file = slots_label_encoder_file
        self.intent_label_encoder_file = intent_label_encoder_file
        self.model_metrics_directory = model_metrics_directory
        self.model_metrics_report_file = model_metrics_report_file
        self.model_confusion_matrix_file = model_confusion_matrix_file
        # Download trained model if needed
        self.__download_trained_model_if_missing(is_training=is_training, is_old=is_old)

        self.tokenizer = FullTokenizer(vocab_file=consts.BERT_VOCAB_FILE_PATH)

        # Read the classes for model
        self.intent_classes = pd.read_csv(consts.INTENT_CLASSES_PATH).intent.tolist()
        self.slots_classes = pd.read_csv(consts.SLOTS_CLASSES_PATH).slots.tolist()
        self.intent_num = len(self.intent_classes)
        self.slots_num = len(self.slots_classes)

        if is_training:
            self.data = self.__load_data()
            self.max_sequence_length = self.data.max_sequence_length
            self.model = self.__create_model()
            self.slots_label_encoder = self.data.slots_label_encoder
            self.intent_label_encoder = self.data.intent_label_encoder
        else:
            self.__load_metadata()
            self.model = self.__load_model()

    def __download_trained_model_if_missing(self, is_training, is_old=False):
        # Only new models will be downloaded. The old versions should be retrained
        if is_old:
            return
        if is_training:
            return

        if not os.path.exists(self.trained_model_directory):
            print("Creating trained model directory!")
            os.mkdir(self.trained_model_directory)

        if not os.path.exists(consts.TRAINED_MODEL_ASSETS_DIRECTORY):
            print("Creating trained model assets directory!")
            os.mkdir(consts.TRAINED_MODEL_ASSETS_DIRECTORY)

        if not os.path.exists(consts.TRAINED_MODEL_VARIABLES_DIRECTORY):
            print("Creating trained model variables directory!")
            os.mkdir(consts.TRAINED_MODEL_VARIABLES_DIRECTORY)

        self.__download_file(self.trained_model_max_sequence_file,
                             consts.TRAINED_MODEL_MAX_SEQUENCE_FILE_DRIVE_ID)
        self.__download_file(self.slots_label_encoder_file,
                             consts.SLOTS_LABEL_ENCODER_FILE_DRIVE_ID)
        self.__download_file(self.intent_label_encoder_file,
                             consts.INTENT_LABEL_ENCODER_FILE_DRIVE_ID)
        self.__download_file(consts.TRAINED_MODEL_SAVED_MODEL_FILE,
                             consts.TRAINED_MODEL_SAVED_MODEL_FILE_DRIVE_ID)
        self.__download_file(consts.TRAINED_MODEL_VOCAB_ASSETS_FILE,
                             consts.TRAINED_MODEL_VOCAB_ASSETS_FILE_DRIVE_ID)
        self.__download_file(consts.TRAINED_MODEL_CHECKPOINT_VARIABLE_FILE,
                             consts.TRAINED_MODEL_CHECKPOINT_VARIABLE_FILE_DRIVE_ID)
        self.__download_file(consts.TRAINED_MODEL_VARIABLE_INDEX_FILE,
                             consts.TRAINED_MODEL_VARIABLE_INDEX_FILE_DRIVE_ID)

    def __download_file(self, file_path, drive_id):
        if not os.path.exists(file_path):
            print("Downloading: " + file_path)
            dm.download_file_from_google_drive(drive_id, file_path)

    def __save_model(self):
        if not os.path.exists(self.trained_model_directory):
            print("Creating trained model directory!")
            os.mkdir(self.trained_model_directory)

        # Save needed metadata
        with open(self.trained_model_max_sequence_file, "w") as f:
            f.write(str(self.max_sequence_length))
        # Save encoders
        with open(self.slots_label_encoder_file, "wb") as f:
            pickle.dump(self.slots_label_encoder, f)
        with open(self.intent_label_encoder_file, "wb") as f:
            pickle.dump(self.intent_label_encoder, f)
        # Save the model
        tf.keras.models.save_model(self.model, self.trained_model_directory)

    def __load_metadata(self):
        if not os.path.exists(self.trained_model_max_sequence_file):
            raise FileNotFoundError("Model metadata file does not exist at path [" +
                                    self.trained_model_max_sequence_file +
                                    "]. There may have been a problem at training or the metadata was not written.")

        if not os.path.exists(self.slots_label_encoder_file):
            raise FileNotFoundError("Model metadata file does not exist at path [" +
                                    self.slots_label_encoder_file +
                                    "]. There may have been a problem at training or the metadata was not written.")

        if not os.path.exists(self.intent_label_encoder_file):
            raise FileNotFoundError("Model metadata file does not exist at path [" +
                                    self.intent_label_encoder_file +
                                    "]. There may have been a problem at training or the metadata was not written.")

        # Load metadata
        with open(self.trained_model_max_sequence_file, "r") as f:
            self.max_sequence_length = int(f.read())

        # Load encoders
        with open(self.slots_label_encoder_file, "rb") as f:
            self.slots_label_encoder = pickle.load(f)
        with open(self.intent_label_encoder_file, "rb") as f:
            self.intent_label_encoder = pickle.load(f)

    def __load_model(self):
        return tf.keras.models.load_model(self.trained_model_directory)

    def __load_data(self):
        return JointIntentSlotFillingData(self.tokenizer,
                                          training_data_path=self.training_data_path,
                                          test_data_path=self.test_data_path)

    def __create_model(self):
        input_ids = Input(shape=(None,), name='input_ids')
        input_mask = Input(shape=(None,), name='input_masks')
        input_segment = Input(shape=(None,), name='segment_ids')
        input_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        bert_inputs = [input_ids, input_mask, input_segment, input_valid_positions]
        bert_pooled_output, bert_sequence_output = BertLayer(n_fine_tune_layer=12, name='BertLayer')(bert_inputs)

        # Add additional layer for intent classification and slot filling
        intents_dropout = Dropout(rate=0.1)(bert_pooled_output)
        intents_fully_connected = Dense(self.intent_num, activation='softmax', name='intent_classifier')(
            intents_dropout)

        slots_dropout = Dropout(rate=0.1)(bert_sequence_output)
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(slots_dropout)
        slots_output = Multiply(name='slots_tagger')([slots_output, input_valid_positions])

        model = Model(inputs=bert_inputs, outputs=[slots_output, intents_fully_connected])

        # compile model
        optimizer = Adam(lr=5e-5)
        losses = {
            'slots_tagger': 'sparse_categorical_crossentropy',
            'intent_classifier': 'sparse_categorical_crossentropy'
        }

        loss_weights = {
            'slots_tagger': 3.0,
            'intent_classifier': 1.0
        }
        metrics = {'intent_classifier': 'acc'}
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        print(model.summary())

        return model

    def __prepare_valid_positions(self, input_valid_positions):
        input_valid_positions = np.expand_dims(input_valid_positions, axis=2)
        input_valid_positions = np.tile(input_valid_positions, (1, 1, self.slots_num))

        return input_valid_positions

    def train(self):
        train_input_ids = self.data.train_input_ids
        train_input_mask = self.data.train_input_masks
        train_segment_ids = self.data.train_segment_ids
        train_valid_positions = self.__prepare_valid_positions(self.data.train_valid_positions)
        train_tags = self.data.train_slots_tokens
        train_intents = self.data.train_intent_tokens

        self.model.fit(x=[train_input_ids, train_input_mask, train_segment_ids, train_valid_positions],
                       y=[train_tags, train_intents],
                       validation_split=0.1,
                       batch_size=16,
                       epochs=10)

        # Test Model
        self.__test(self.data)
        self.__save_model()

    def __test(self, data):
        test_predicted_intents = []
        test_predicted_slots = []
        test_input_texts = data.test_input_data.tolist()
        for text in test_input_texts:
            intent, slots = self.predict(text)
            test_predicted_intents.append(intent)
            test_predicted_slots.append(slots)

        # print("Classification Report for intents")
        intent_classification_report = classification_report(data.test_intent_data,
                                                             test_predicted_intents)
        # print(intent_classification_report)

        # print("F1-Score for slots")
        test_slots_data = [el.split() for el in data.test_slots_data]
        slots_f1_score = f1_score(list(chain.from_iterable(test_slots_data)),
                                  list(chain.from_iterable(test_predicted_slots)),
                                  average='micro')

        # print("Slots f1-score:", slots_f1_score)
        if not os.path.exists(self.model_metrics_directory):
            print("Creating trained model metrics directory!")
            os.mkdir(self.model_metrics_directory)

        with open(self.model_metrics_report_file, 'w') as f:
            f.write("Classification Report for intents:\n\n\n")
            f.write(intent_classification_report)
            f.write('\n\n\n')
            f.write('Slots f1-score: ')
            f.write(str(slots_f1_score))

        # Plot confusion matrix and save the plot
        cm = confusion_matrix(data.test_intent_data, test_predicted_intents, labels=self.intent_classes)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title("Confusion Matrix")
        fig.colorbar(cax)
        ax.set_xticklabels([''] + self.intent_classes)
        ax.set_yticklabels([''] + self.intent_classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(self.model_confusion_matrix_file)

    def predict(self, data):
        input_token_ids, input_masks, input_segments, input_valid_positions, valid_positions = \
            self.__prepare_data(data)
        slots, intents = self.model.predict([input_token_ids, input_masks, input_segments, input_valid_positions])

        slots = self.__slots_inverse_transform(slots, valid_positions)
        # Delete first and last elements (presumably '[CLS]' and '[SEP]')
        slots = np.array([x[1:-1] for x in slots])

        intents = np.array(
            [self.intent_label_encoder.inverse_transform([np.argmax(intents[i])])[0] for i in range(intents.shape[0])])

        return intents[0], slots[0]

    def __slots_inverse_transform(self, model_output, valid_positions):
        slots = np.argmax(model_output, axis=-1)
        slots = [self.slots_label_encoder.inverse_transform(y) for y in slots]
        output = []
        for i in range(len(slots)):
            y = []
            for j in range(self.max_sequence_length):
                if valid_positions[i][j] == 1:
                    y.append(str(slots[i][j]))
            output.append(y)

        return output

    def __prepare_data(self, text):
        input_token_ids, input_masks, input_segments, input_valid_positions = self.__tokenize_input_text(text)

        return input_token_ids, \
               input_masks, \
               input_segments, \
               self.__prepare_valid_positions(input_valid_positions), \
               input_valid_positions

    def __tokenize_input_text(self, text):
        # TODO: Maybe merge the 2 tokenizations or something
        text = text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
        tokens = self.tokenizer.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_masks = [1] * len(input_token_ids)
        input_segments = [0] * len(tokens)
        input_valid_positions = [0 if "##" in el else 1 for el in tokens]

        input_token_ids = tf.keras.preprocessing.sequence.pad_sequences([input_token_ids],
                                                                        maxlen=self.max_sequence_length,
                                                                        truncating='post',
                                                                        padding='post')

        input_masks = tf.keras.preprocessing.sequence.pad_sequences([input_masks],
                                                                    maxlen=self.max_sequence_length,
                                                                    truncating='post',
                                                                    padding='post')

        input_segments = tf.keras.preprocessing.sequence.pad_sequences([input_segments],
                                                                       maxlen=self.max_sequence_length,
                                                                       truncating='post',
                                                                       padding='post')

        input_valid_positions = tf.keras.preprocessing.sequence.pad_sequences([input_valid_positions],
                                                                              maxlen=self.max_sequence_length,
                                                                              truncating='post',
                                                                              padding='post')

        return np.array(input_token_ids), \
               np.array(input_masks), \
               np.array(input_segments), \
               np.array(input_valid_positions)


class BertLayer(Layer):
    def __init__(self,
                 n_fine_tune_layer=12,
                 bert_path='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',
                 **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layer
        self.trainable = True
        self.output_size = 768
        self.bert_path = bert_path
        super(BertLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super(BertLayer, self).get_config().copy()
        config.update({
            'n_fine_tune_layers': self.n_fine_tune_layers,
            'trainable': True,
            'output_size': self.output_size,
            'bert_path': self.bert_path
        })

        return config

    def build(self, input_shape):
        self.bert = hub.KerasLayer(self.bert_path, trainable=self.trainable, name="{}_module".format(self.name))
        trainable_vars = self.bert.variables
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if "/cls/" not in var.name]
        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        # Add non_trainable weights:

        # Workaround ...
        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars if "/cls/" in var.name]
        for var in trainable_vars:
            self._non_trainable_weights.append(var)

        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars if "/cls/" not in var.name]
        trainable_vars = trainable_vars[:len(trainable_vars) - self.n_fine_tune_layers]
        for var in trainable_vars:
            self._non_trainable_weights.append(var)

        # This piece of code doesn't work due to not being able to compare 2 tf.Variable types
        # The error seem to be coming from numpy
        # for var in self.bert.variables:
        #     if var not in self._trainable_weights:
        #         self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]  # cast the variables to int32 tensor
        input_ids, input_mask, segment_ids, valid_positions = inputs
        pooled_output, sequence_output = self.bert([input_ids, input_mask, segment_ids])
        return pooled_output, sequence_output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_size
