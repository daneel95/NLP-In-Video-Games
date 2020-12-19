import os
import intent.download_models as dm
import intent.constants as consts

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from sklearn.metrics import classification_report

from intent.intent_classification_data import IntentClassificationData


class IntentClassificationModel:
    def __init__(self, is_training=False):
        # Prepare BERT paths
        self.bert_checkpoint_dir = consts.BERT_CHECKPOINT_DIRECTORY
        self.bert_checkpoint_file = os.path.join(self.bert_checkpoint_dir, 'bert_model.ckpt')
        self.bert_config_file = os.path.join(self.bert_checkpoint_dir, 'bert_config.json')

        # Downlaod models if needed
        self.__download_bert_model_if_missing()
        self.__download_trained_model_if_missing(is_training)

        # create tokenizer from bert vocab
        self.tokenizer = FullTokenizer(vocab_file=os.path.join(self.bert_checkpoint_dir, 'vocab.txt'))

        # Classes of the model, hard coded to not always load the data
        self.classes = ["AnswerQuestion", "FollowAction"]

        # if training then calculate max_sequence_length from data and save it alongside model
        if is_training:
            self.data = self.__load_data()
            self.max_sequence_length = self.data.max_sequence_length
            self.model = self.__create_model()
        # if not training then just load the model and max_sequence_length
        else:
            self.__load_max_sequence()
            self.model = self.__create_model()
            self.__load_model()

    def __download_bert_model_if_missing(self):
        if not os.path.exists(consts.BERT_CHECKPOINT_DIRECTORY):
            print("Creating BERT model directory!")
            os.mkdir(consts.BERT_CHECKPOINT_DIRECTORY)
        if not os.path.exists(consts.BERT_CONFIG_FILE):
            print("Downloading: " + consts.BERT_CONFIG_FILE)
            dm.download_file_from_google_drive(consts.BERT_CONFIG_FILE_DRIVE_ID,
                                               consts.BERT_CONFIG_FILE)
        if not os.path.exists(consts.BERT_FULL_CHECKPOINT_FILE):
            print("Downloading: " + consts.BERT_FULL_CHECKPOINT_FILE)
            dm.download_file_from_google_drive(consts.BERT_FULL_CHECKPOINT_FILE_DRIVE_ID,
                                               consts.BERT_FULL_CHECKPOINT_FILE)
        if not os.path.exists(consts.BERT_MODEL_CHEKPOINT_INDEX_FILE):
            print("Downloading: " + consts.BERT_MODEL_CHEKPOINT_INDEX_FILE)
            dm.download_file_from_google_drive(consts.BERT_MODEL_CHECKPOINT_INDEX_FILE_DRIVE_ID,
                                               consts.BERT_MODEL_CHEKPOINT_INDEX_FILE)
        if not os.path.exists(consts.BERT_MODEL_CHECKPOINT_META_FILE):
            print("Downloading: " + consts.BERT_MODEL_CHECKPOINT_META_FILE)
            dm.download_file_from_google_drive(consts.BERT_MODEL_CHECKPOINT_META_FILE_DRIVE_ID,
                                               consts.BERT_MODEL_CHECKPOINT_META_FILE)
        if not os.path.exists(consts.BERT_MODEL_CHECKPOINT_VOCAB_FILE):
            print("Downloading: " + consts.BERT_MODEL_CHECKPOINT_VOCAB_FILE)
            dm.download_file_from_google_drive(consts.BERT_MODEL_CHECKPOINT_VOCAB_FILE_DRIVE_ID,
                                               consts.BERT_MODEL_CHECKPOINT_VOCAB_FILE)

    def __download_trained_model_if_missing(self, is_training):
        if not os.path.exists(consts.TRAINED_MODEL_DIRECTORY):
            print("Creating trained model directory!")
            os.mkdir(consts.TRAINED_MODEL_DIRECTORY)

        if is_training:
            return

        if not os.path.exists(consts.TRAINED_MODEL_WEIGHTS_FILE):
            print("Downloading: " + consts.TRAINED_MODEL_WEIGHTS_FILE)
            dm.download_file_from_google_drive(consts.TRAINED_MODEL_WEIGHTS_FILE_DRIVE_ID,
                                               consts.TRAINED_MODEL_WEIGHTS_FILE)

        if not os.path.exists(consts.TRAINED_MODEL_MAX_SEQUENCE_FILE):
            print("Downloading: " + consts.TRAINED_MODEL_MAX_SEQUENCE_FILE)
            dm.download_file_from_google_drive(consts.TRAINED_MODEL_MAX_SEQUENCE_FILE_DRIVE_ID,
                                               consts.TRAINED_MODEL_MAX_SEQUENCE_FILE)

    def __create_model(self):
        with tf.io.gfile.GFile(self.bert_config_file, 'r') as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(bert_params, name='bert')

        input_ids = keras.layers.Input(shape=(self.max_sequence_length,), dtype='int32', name='input_ids')
        bert_output = bert(input_ids)

        print('Bert shape:', bert_output.shape)

        cls_out = keras.layers.Lambda(lambda sequence: sequence[:, 0, :])(bert_output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)

        logits = keras.layers.Dense(units=bert_output.shape[2], activation='tanh')(cls_out)
        logits = keras.layers.Dropout(0.5)(logits)
        logits = keras.layers.Dense(units=len(self.classes), activation='softmax')(logits)

        model = keras.Model(inputs=input_ids, outputs=logits)
        model.build(input_shape=(None, self.max_sequence_length))

        load_stock_weights(bert, self.bert_checkpoint_file)

        model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])

        print(model.summary())
        return model

    def __load_max_sequence(self):
        if not os.path.exists(consts.TRAINED_MODEL_MAX_SEQUENCE_FILE):
            raise FileNotFoundError("Model max sequence file does not exist at path [" +
                                    consts.TRAINED_MODEL_MAX_SEQUENCE_FILE +
                                    "]. There may have been a problem at training or the model was not trained.")

        with open(consts.TRAINED_MODEL_MAX_SEQUENCE_FILE, "r") as f:
            self.max_sequence_length = int(f.read())

    def __load_model(self):
        self.model.load_weights(consts.TRAINED_MODEL_WEIGHTS_FILE)

    def __save_model(self):
        with open(consts.TRAINED_MODEL_MAX_SEQUENCE_FILE, "w") as f:
            f.write(str(self.max_sequence_length))
        self.model.save_weights(consts.TRAINED_MODEL_WEIGHTS_FILE)

    def train(self):
        log_dir = 'log/intent_classification_logs'
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        history = self.model.fit(x=self.data.train_x,
                                 y=self.data.train_y,
                                 validation_split=0.1,
                                 batch_size=4,  # 16
                                 shuffle=True,
                                 epochs=5,
                                 callbacks=[tensorboard_callback])

        # also print some stats about the model (test)
        self.__test(self.data)

        self.__save_model()

    def __load_data(self):
        training_data = pd.read_csv(consts.TRAINING_DATA_PATH)
        test_data = pd.read_csv(consts.TEST_DATA_PATH)
        return IntentClassificationData(training_data, test_data, self.tokenizer, self.classes, max_sequence_length=128)

    # single text prediction
    def predict(self, text):
        data = self.__prepare_data(text)
        prediction = self.model.predict(data).argmax(axis=-1)

        return prediction[0], self.classes[prediction[0]]

    def __prepare_data(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        cut_point = min(len(token_ids), self.max_sequence_length - 2)
        token_ids = token_ids[:cut_point]
        token_ids = token_ids + [0] * (self.max_sequence_length - len(token_ids))

        return np.array([token_ids])

    def __test(self, data):
        _, train_accuracy = self.model.evaluate(data.train_x, data.train_y)
        _, test_accuracy = self.model.evaluate(data.test_x, data.test_y)
        print("Train accuracy is:", train_accuracy)
        print("Test accuracy is:", test_accuracy)

        y_predictions_test = self.model.predict(data.test_x).argmax(axis=-1)
        print(classification_report(data.test_y, y_predictions_test, target_names=self.classes))
