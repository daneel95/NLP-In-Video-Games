from question_answering.question_answering import QuestionAnsweringModel

from flask import Flask
from flask import request

app = Flask(__name__)
model = None
# This is the context where the answers will come from.
# It can be fetched from a DB or any other place
# Or it can come from a http post request
# For thesis purposes, I hardcoded the whole context.
TELEPORTER_CONTEXT = "Teleporter teleports you to the desired location. " \
                     "Teleporter lets you choose different locations to teleport to. " \
                     "The teleporter will cost you 10 gold for each teleportation. " \
                     "In order to use the teleporter just interact with it and choose a location. " \
                     "Teleporter will also help you teleport to other big cities in the game and dungeon entrances."
HEALING_SHRINE_CONTEXT = "Healing Shrine heals your HP to maximum. " \
                         "If you need to prepare for a battle then Healing Shrine is the way to go. " \
                         "Healing Shrine also gives you different buffs:" \
                         " strength, attack power and ability power buffs which cost 1 gold each buff, " \
                         "so they are not free."
WEAPON_SHOP_CONTEXT = "You can get better gear (weapons and armors) at Weapon Shop and Armor Shop in every city. " \
                      "You can buy weapons at the Weapon Shop. " \
                      "Another reason to visit the Weapon Shop is upgrading your weapons. " \
                      "Each weapon upgrade require to pay a fee depending on the level of your weapon and the" \
                      " level you want to enhance to. By upgrading your weapons, you get more damage and more stats."
ARMOR_SHOP_CONTEX = "Armor Shop is a place where you can buy new armors and upgrade armors. " \
                    "Each armor upgrade requires you to pay a fee depending on the level of " \
                    "your armor and the level you want to enhance to. " \
                    "The benefit of upgrading your armor is that you become more tanky and will die slower."
POTION_SHOP_CONTEXT = "Mana Potions, Health Potions and Buff Potions can be bought at the Potion Shop. " \
                      "The Potion Shop can be found in any major city, it just sells potions so the player " \
                      "can become stronger and survive in fights. Every potion costs 20 silver a piece."
DEEP_SEA_DUNGEON_ENTRANCE_CONTEXT = "Deep Sea Dungeon is the hardest dungeon in the game. " \
                                    "Deep Sea Dungeon requires a party of at least 4 people and at least level 50. " \
                                    "Deep Sea Dungeon has 3 bosses: Sea Monster, " \
                                    "Atlantis General and Atlan: King of Atlantis. " \
                                    "Deep Sea Dungeon will drop the legendary item Spear of Atlan, " \
                                    "the best weapon in the game."
CASTLE_OF_DOOM_DUNGEON_ENTRANCE_CONTEXT = "Castle of Doom dungeon requires a party of 3 people and at least level " \
                                          "30. Castle of Doom dungeon has only one boss, Frankenstein. " \
                                          "In order to complete Castle of Doom dungeon you need to pass " \
                                          "the trial of the dungeon. " \
                                          "Castle of Doom dungeon drops the legendary item Doom Helmet."
DRAGON_DUNGEON_ENTRANCE_CONTEXT = "Dragon Dungeon is a solo dungeon and requires at least level 50. " \
                                  "Dragon dungeon has 20 bosses where you are required to defeat the previous " \
                                  "boss to advance to the next level. Dragon Dungeon's last boss is " \
                                  "God of Dragons: Ignius and he drops the legendary item Chest of Fire," \
                                  " which is the strongest item in the game. In order to enter Dragon Dungeon you " \
                                  "need the ticket of dragon which can be farmed at high level monster in the world."
TUTORIAL_DUNGEON_ENTRANCE_CONTEXT = "Tutorial Dungeon is a dungeon where you can learn the basics of" \
                                    " fighting, tanking, healing and game mechanics."

CONTEXT = TELEPORTER_CONTEXT + " \n\n" + \
          HEALING_SHRINE_CONTEXT + " \n\n" + \
          WEAPON_SHOP_CONTEXT + " \n\n" + \
          ARMOR_SHOP_CONTEX + " \n\n" + \
          POTION_SHOP_CONTEXT + " \n\n" + \
          DEEP_SEA_DUNGEON_ENTRANCE_CONTEXT + " \n\n" + \
          CASTLE_OF_DOOM_DUNGEON_ENTRANCE_CONTEXT + " \n\n" + \
          DRAGON_DUNGEON_ENTRANCE_CONTEXT + " \n\n" + \
          TUTORIAL_DUNGEON_ENTRANCE_CONTEXT + " \n\n"

# Answer to question according to context
@app.route("/answer", methods=['GET'])
def answer():
    question = request.args['question']

    if model:
        return model.answer(question=question)[0]

    return "I don't know!"


# Change the context
@app.route("/context", methods=['POST'])
def create_context():
    request_data = request.get_json()
    global CONTEXT
    CONTEXT = request_data['context']
    global model
    model = QuestionAnsweringModel(context=CONTEXT)


if __name__ == "__main__":
    # Create question answering model
    model = QuestionAnsweringModel(context=CONTEXT)
    # Run the server
    app.run(debug=True, port=5001)
