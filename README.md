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


## Examples

### Question Answering

1. Where can i heal?
   
![First Question](Demo/FirstQuestion.gif)

2. which is th hardest dungeon in th game?

![Second Question](Demo/SecondQuestion.gif)

3. doom dungeon required level

![Third Question](Demo/ThirdQuestion.gif)

### Follow Action

1. Get me to teleportr

![First Follow](Demo/FirstFollow.gif)

2. Get me to arm shp

![Second Follow](Demo/SecondFollow.gif)

3. show me castle of doom entrance

![Third Follow](Demo/ThirdFollow.gif)

## References

- [Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", October 2018](https://arxiv.org/pdf/1810.04805v2.pdf)
- [Qian Chen, Zhu Zhuo, Wen Wang, "BERT for Joint Intent Classification and Slot Filling",  February 2019](https://arxiv.org/pdf/1902.10909.pdf)
- [OpenAI Team, "Language Models are Few-Shot Learners", May 2020](https://arxiv.org/pdf/2005.14165.pdf) 
- ["joint-intent-classification-and-slot-filling-based-on-BERT"](https://github.com/90217/joint-intent-classification-and-slot-filling-based-on-BERT)
- [Moscow Institute of Physics and Technology (MIPT), "DeepPavlov"](https://github.com/deepmipt/DeepPavlov) 

