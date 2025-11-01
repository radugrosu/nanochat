"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc
"""

from typing import Literal, cast

from datasets import Dataset, load_dataset

from nanochat.tokenizer import Conversation, Message
from tasks.common import Task, render_mc


class ArcConversation(Conversation):
    letters: list[str]


class ARC(Task):
    def __init__(self, subset: Literal["ARC-Easy", "ARC-Challenge"], split: str, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], (
            "ARC subset must be ARC-Easy or ARC-Challenge"
        )
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)
        self.ds = cast(Dataset, ds)

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> Conversation:
        row = self.ds[index]
        question = row["question"]  # the question text
        choices = row["choices"]["text"]  # the text of each choice
        answer_string = row["answerKey"]  # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"]  # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, (
            f"ARC answer {answer_string} must be one of {letters}"
        )  # sanity check
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages: list[Message] = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string},
        ]
        conversation: ArcConversation = {
            "messages": messages,
            "letters": letters,  # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation: ArcConversation, response: str) -> bool:
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert response in conversation["letters"], (
            f"ARC answer {response} is expected to be one of {conversation['letters']}"
        )
        assistant_message = conversation["messages"][-1]["content"]  # e.g. "A"
        return response == assistant_message
