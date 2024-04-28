import os
import json
from tqdm import tqdm
from example_freature import MRCExample


class SquadProcessor():
    """
    Processor for the SQuAD-style data set.
    """
    def __init__(self):
        
        self.type_list = ['None', 'Business.Declare-Bankruptcy', 'Business.End-Org', 'Business.Merge-Org', 'Business.Start-Org', 'Conflict.Attack', 'Conflict.Demonstrate', 'Contact.Meet', 'Contact.Phone-Write', 'Justice.Acquit', 'Justice.Appeal', 'Justice.Arrest-Jail', 'Justice.Charge-Indict', 'Justice.Convict', 'Justice.Execute', 'Justice.Extradite', 'Justice.Fine', 'Justice.Pardon', 'Justice.Release-Parole', 'Justice.Sentence', 'Justice.Sue', 'Justice.Trial-Hearing', 'Life.Be-Born', 'Life.Die', 'Life.Divorce', 'Life.Injure', 'Life.Marry', 'Movement.Transport', 'Personnel.Elect', 'Personnel.End-Position', 'Personnel.Nominate', 'Personnel.Start-Position', 'Transaction.Transfer-Money', 'Transaction.Transfer-Ownership']

        self.event2id = dict([(type,i) for i, type in enumerate(self.type_list)])

    def get_examples(self, data_dir, filename, mode):
        if data_dir is None:
            data_dir = ""
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, set_type=mode)
    
    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False
                    event_type = paragraph['type'].split("_")[0]
                    labels = self.event2id[event_type]

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                            if answer_text[0] != context_text[start_position_character]:
                                # print("%s -> %s"%(answer_text, context_text[start_position_character:start_position_character+len(answer_text)]))
                                new_posion = context_text.find(answer_text)
                                if new_posion >= 0:
                                    start_position_character = new_posion
                                # print(start_position_character, new_posion)
                        else:
                            answers = qa["answers"]

                    example = MRCExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                        labels=labels,
                    )

                    examples.append(example)
        return examples


class MRCJsonlProcessor():
    def __init__(self):
        self.event2id = {}
        self.event_id = 0
        
    
    def get_examples(self, data_dir, filename, mode):
        if data_dir is None:
            data_dir = ""
        examples = []
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as reader:
            for line in reader:
                example = json.loads(line)
                examples.append(example)
        print(self.event2id)
        return self._create_examples(examples, set_type=mode)

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        
        for entry in tqdm(input_data):
            title = entry["title"]
            qas_id = entry["id"]
            question_text = entry['question']
            context_text = entry["context"]

            start_position_character = None
            answer_text = None
            answers = entry['answers']
            is_impossible = False
            
            if 'is_impossible' in entry:
                is_impossible = entry["is_impossible"]
            if not is_impossible and len(entry["answers"]) > 0:
                answer = entry["answers"][0]
                answer_text = answer["text"]
                start_position_character = answer["answer_start"]
            event_type = entry['event_argument'].split("_")[0]
            if event_type not in self.event2id:
                self.event2id[event_type] = self.event_id
                self.event_id += 1
            labels = self.event2id[event_type]
        
            example = MRCExample(
                qas_id=qas_id,
                question_text=question_text,
                context_text=context_text,
                answer_text=answer_text,
                start_position_character=start_position_character,
                title=title,
                is_impossible=is_impossible,
                answers=answers,
                labels=labels,
            )

            examples.append(example)           
        return examples
