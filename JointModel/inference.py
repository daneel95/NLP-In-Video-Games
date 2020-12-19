from joint_intent_slot_filling.joint_model import JointIntentClassificationSlotsFillingModel
from joint_intent_slot_filling.location_extractor import LocationExtractor
from joint_intent_slot_filling.constants import INFERENCE_LOGS_FILE_LOCATION

from flask import Flask, request, jsonify
import string
import requests
app = Flask(__name__)

model = None
LOCATION_BEGINNING_SLOT = 'B-location'
LOCATION_INTERIOR_SLOT = 'I-location'

QUESTION_ANSWERING_HOST = "http://127.0.0.1"
QUESTION_ANSWERING_PORT = "5001"
QUESTION_ANSWERING_PATH = "/answer"

# Location titles strings from the game
LOCATIONS = []


class ActionType:
    ANSWER_QUESTION = "AnswerQuestion"
    FOLLOW_ACTION = "FollowAction"


@app.route("/inference-old", methods=['GET'])
def old_inference():
    text = request.args["text"]
    prediction_id, prediction_class = model.predict(text)

    # For now just return an answer.
    # TODO: Properly handle each type differently
    if prediction_class == "AnswerQuestion":
        return "Answering the question according to given text: [" + text + "]"
    elif prediction_class == "FollowAction":
        return "Doing a 'Follow me' action according to given text: [" + text + "]"
    return "Failed to predict any known action!!"


def __find_locations(text, slots):
    locations = []
    # TODO: Also handle here if not skipping punctuation. For now punctuation is skipped
    text = text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    text_words = text.split()
    current_location = ''
    for word, slot in zip(text_words, slots):
        if slot == LOCATION_BEGINNING_SLOT:
            if current_location != '':
                locations.append(current_location)
            current_location = word
        elif slot == LOCATION_INTERIOR_SLOT:
            if current_location == '':
                continue
            current_location += " " + word
    if current_location != '':
        locations.append(current_location)

    return locations


def __answer_question_response(text):
    response = dict()
    response["answer"] = __answer_question(question=text)
    response["action"] = ActionType.ANSWER_QUESTION

    return response


def __follow_action_response(text, slots):
    locations = __find_locations(text=text, slots=slots)
    response = dict()
    # Couldn't extract any location from it
    if len(locations) == 0:
        response["answer"] = "Didn't catch that!"
    else:
        location_extractor = LocationExtractor(LOCATIONS)
        most_similar_location = location_extractor.get_most_similar_location_sequence_matcher(locations[0])
        if most_similar_location is None:
            response["answer"] = "Can't find the requested location!"
        else:
            response["answer"] = "Follow me!"
            response["location"] = most_similar_location
    response["action"] = ActionType.FOLLOW_ACTION

    return response


def __bad_response():
    return jsonify({
        "answer": "Didn't catch that!",
        "action": "Unknown"
    })


def __normalize_string(str_value):
    return str_value.lower().replace(" ", "")


def __log_response(input_text, intent, slots, response):
    with open(INFERENCE_LOGS_FILE_LOCATION, 'a') as f:
        f.write("Input Text: " + input_text + "\n")
        f.write("Predicted Intent: " + intent + "\n")
        f.write("Predicted Slots: [" + ', '.join(slots) + "]\n")
        if intent == 'AnswerQuestion':
            f.write("Predicted Question Answer: " + response["answer"] + "\n")
        elif intent == 'FollowAction':
            if response["answer"] == "Didn't catch that!":
                f.write("Predicted Location: Didn't find any SLOTS\n")
            elif response["answer"] == "Can't find the requested location!":
                f.write("Predicted Location: SLOTS were found but no proper location could be chosen!!\n")
            else:
                f.write("Predicted Location: " + str(response["location"]) + "\n")

        f.write("==========================\n\n\n")


@app.route("/inference", methods=['GET'])
def inference():
    text = request.args["text"]
    intent, slots = model.predict(text)

    if intent == 'AnswerQuestion':
        response = __answer_question_response(text=text)
        __log_response(input_text=text, intent=intent, slots=slots, response=response)
        return jsonify(response)
    elif intent == 'FollowAction':
        response = __follow_action_response(text=text, slots=slots)
        __log_response(input_text=text, intent=intent, slots=slots, response=response)
        return jsonify(response)

    return __bad_response()


@app.route("/locations", methods=['POST'])
def set_locations():
    global LOCATIONS
    LOCATIONS = request.json["locations"]
    return "Success"


def __answer_question(question):
    question_answering_endpoint = QUESTION_ANSWERING_HOST + ":" + QUESTION_ANSWERING_PORT + QUESTION_ANSWERING_PATH
    params = {"question": question}
    response = requests.get(question_answering_endpoint, params=params)
    if response.status_code != 200:
        return "I don't know"

    return response.text


if __name__ == "__main__":
    # Create the model
    model = JointIntentClassificationSlotsFillingModel(is_training=False)
    # Run the server
    app.run(debug=True)
