"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import json
from pathlib import Path
from typing import cast

from nanochat.common import FilePath
from nanochat.tokenizer import Conversation, Message
from tasks.common import Task


def validate_conversation(messages_data: list[dict[str, str]]) -> list[Message]:
    """Validate conversation using Pydantic with additional structure checks"""
    if len(messages_data) < 2:
        raise ValueError("Conversation must have at least 2 messages")

    # Validate each message
    assert isinstance(messages_data, list), f"Expected list of messages, got {type(messages_data)}"
    assert len(messages_data) >= 2, (
        f"Conversation must have at least 2 messages, got {len(messages_data)}"
    )

    # Validate alternating roles
    for i, message in enumerate(messages_data):
        assert "role" in message, f"Message {i} missing 'role' field"
        assert "content" in message, f"Message {i} missing 'content' field"
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert message["role"] == expected_role, (
            f"Message {i} has role {message['role']} but should be {expected_role}"
        )
        assert isinstance(message["content"], str), f"Message {i} content must be a string"
    messages = cast(list[Message], messages_data)
    return messages


class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath: FilePath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations: list[list[Message]] = []

        # Load all conversations from the JSONL file
        if not Path(filepath).exists():
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print(
                "If you recently did a git pull and suddely see this, it might be due to the new addition of identity conversations"
            )
            print(
                "See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139"
            )
            print(
                "Quick fix: simply run the following command to download the file and you're done:"
            )
            print(
                f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
            )
            print("-" * 80)

        else:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    messages_data = json.loads(line)
                    messages = validate_conversation(messages_data)
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index: int) -> Conversation:
        messages = self.conversations[index]
        conversation: Conversation = {
            "messages": messages,
        }
        return conversation
