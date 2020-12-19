from deeppavlov import build_model, configs
import subprocess


class QuestionAnsweringModel:
    def __init__(self, context):
        with open('question_answering/should_download.txt', 'r') as f:
            should_download = bool(f.read())
        self.model = build_model(configs.squad.squad, download=should_download)
        self.context = context

        if should_download:
            with open('question_answering/should_download.txt', 'w') as f:
                f.write("False")

    def __install_dependencies(self):
        dependencies_command = 'python -m deeppavlov install squad_bert'
        process = subprocess.Popen(dependencies_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print(output)
        print(error)

    def answer(self, question):
        return self.model([self.context], [question])[0]


if __name__ == "__main__":
    test_context = 'DeepPavlov is library for NLP and dialog systems.'
    test_question = 'What is DeepPavlov?'
    question_answering_model = QuestionAnsweringModel(context=test_context)
    answer = question_answering_model.answer(question=test_question)
    print(answer)
