from joint_intent_slot_filling.joint_model import JointIntentClassificationSlotsFillingModel
import joint_intent_slot_filling.constants as consts

# Trains or retrains the model
if __name__ == "__main__":
    # Training
    joint_model = JointIntentClassificationSlotsFillingModel(is_training=True,
                                                             training_data_path=consts.OLD_TRAINING_DATA_PATH,
                                                             test_data_path=consts.OLD_TEST_DATA_PATH,
                                                             trained_model_directory=consts.OLD_TRAINED_MODEL_DIRECTORY,
                                                             trained_model_max_sequence_file=consts.OLD_TRAINED_MODEL_MAX_SEQUENCE_FILE,
                                                             slots_label_encoder_file=consts.OLD_SLOTS_LABEL_ENCODER_FILE,
                                                             intent_label_encoder_file=consts.OLD_INTENT_LABEL_ENCODER_FILE,
                                                             model_metrics_directory=consts.MODEL_METRICS_DIRECTORY,
                                                             model_metrics_report_file=consts.OLD_MODEL_METRICS_REPORT_FILE,
                                                             model_confusion_matrix_file=consts.OLD_MODEL_CONFUSION_MATRIX_FILE,
                                                             is_old=True)
    joint_model.train()
