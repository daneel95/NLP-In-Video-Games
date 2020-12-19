# NLP In Video Games

This repository contains the implementation of NLP In Video games created in order to test its functionality.

The repository contains the following:
- Test Game
- Joint Model for intent classification and slots filling
- Question Answering model used to answer questions

## Test Game

The test game is a simple simulation in which several locations were created. Those locations are used in order to test the NLP models.

The game contains a simple chat system used to communicate with the BOT using Natural Language.

In order to run the game **Unreal Engine 4.25** is required. The game can run either from the engine itself or can be built to test it.

## Joint Model

Joint model is a model which, from a given text, finds out what the intent of the user is and what the slots in the input text are.

For more information about the Joint Model see `JointModel` directory.

## Question Answering

Question Answering is a model which answers to the given question sent as a text.

For more information about the Question Answering Model see `QuestionAnswering` directory.
