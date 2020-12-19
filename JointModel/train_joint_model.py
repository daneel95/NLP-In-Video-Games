from joint_intent_slot_filling.joint_model import JointIntentClassificationSlotsFillingModel

# Trains or retrains the model
if __name__ == "__main__":
    # Training
    joint_model = JointIntentClassificationSlotsFillingModel(is_training=True)
    joint_model.train()
