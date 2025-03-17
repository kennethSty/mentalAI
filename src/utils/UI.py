import datetime
import sys
from typing import List, Optional, Any
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

NEW_LINE = "\n"
SEPARATOR = "_" * 120
INDENTATION = "      "
LINE = INDENTATION + SEPARATOR
FRAME_LINE = LINE + NEW_LINE
LOGO = (f"{INDENTATION}       |\n"
        f"{INDENTATION}     \\\|//\n"
        f"{INDENTATION}     \\\|//\n"
        f"{INDENTATION}    \\\\\|///\n"
        f"{INDENTATION}    \\\\\|///\n"
        f"{INDENTATION}     \\\|//\n"
        f"{INDENTATION}      \|/\n"
        f"{INDENTATION}       |\n")

class PrettyStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self, max_tokens_per_line=26, whitespace=INDENTATION):
        super().__init__()
        self.max_tokens_per_line = max_tokens_per_line
        self.token_count = 0
        self.whitespace = whitespace
        self.first_token = True
        self.previous_token = ""
        self.consecutive_newlines = 0  # Track empty lines
        self.max_empty_lines = 2

    def on_llm_new_token(self, token: str, **kwargs):
        is_eol = (token == "\n")
        punctuation_set = {" ", ",", ".", ";", ":"}
        is_whitespace_or_eos = token in punctuation_set
        previous_is_whitespace_or_eos = self.previous_token in punctuation_set

        if self.first_token:
            if is_eol:
                print(f"{self.whitespace}", end="")
            else:
                print(f"{self.whitespace}{token}", end="")
            self.first_token = False

        elif self.token_count > self.max_tokens_per_line:
            self.token_count = 0
            if is_whitespace_or_eos:
                print(f"{token}", end="")
            elif previous_is_whitespace_or_eos:
                print(f"\n{self.whitespace}{token}", end="")
            else:
                print(f"\n{self.whitespace}{token}", end="")

        else:
            if is_eol:
                self.token_count = 0
                self.consecutive_newlines += 1
                if self.consecutive_newlines >= self.max_empty_lines:
                    return
                else:
                    print(f"\n{self.whitespace}", end="")
            else:
                self.consecutive_newlines = 0
                print(token, end="")

        self.token_count += 1


def print_greeting() -> None:
    print(LOGO)
    print(LINE)
    print(f"{INDENTATION}Hi I am HealthMate and happy to support you to be at your best")
    print(f"{INDENTATION}What is on your mind lately!")
    print(LINE)

def print_farewell() -> None:
    print(f"{INDENTATION}Stay healthy!")
    print(LINE)

def print_separator() -> None:
    print(LINE)

def print_newline_separator() -> None:
    print(NEW_LINE)
    print(LINE)
