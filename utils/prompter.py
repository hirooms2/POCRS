import json
import os
import random
from copy import deepcopy
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose", "args")

    def __init__(self, args, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        self.args = args
        file_name = os.path.join(args.home, "templates", f"{template_name}.json")
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_instructions(self,
                              mode,
                              dataset_input: list,
                              dataset_output: list):
        instructions = []

        for data, label in zip(dataset_input, dataset_output):

            predicted_goal = data['predicted_goal'][0]

            topk_topic = self.args.topk_topic
            predicted_topic_list = deepcopy(data['predicted_topic'][:topk_topic])

            if 'D2I' == self.args.prompt or 'D2I_cot' == self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))
            elif 'D2R' in self.args.prompt:
                instructions.append(self.generate_prompt(instruction=data['dialog'], label=label, mode=mode))

        instructions = [i.replace('\xa0', ' ').replace('  ', ' ').strip() for i in instructions]
        return instructions

    def generate_prompt(
            self,
            instruction: str,
            input: Union[None, str] = None,
            input2: Union[None, str] = None,
            input3: Union[None, str] = None,
            input4: Union[None, str] = None,
            label: Union[None, str] = None,
            mode: str = 'test') -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input and not input2:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        elif input and input2 and not input3:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2
            )
        elif input and input2 and input3 and not input4:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2, input3=input3
            )
        elif input and input2 and input3 and input4:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input, input2=input2, input3=input3, input4=input4
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if mode == 'train':
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
