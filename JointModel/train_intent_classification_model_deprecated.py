from intent.intent_classification_model import IntentClassificationModel

# Run the main to train the model (or to retrain the model)
if __name__ == "__main__":
    # training
    intent_classification_model = IntentClassificationModel(is_training=True)
    intent_classification_model.train()
