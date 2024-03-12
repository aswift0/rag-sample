from llama_index.core.tools import FunctionTool
import os

note = os.path.join("data", "notes.txt")

def save_note(note):
    if not os.path.exists:
        open(note, "w")

    with open(note, "a") as file:
        file.writelines([note+"\n"])

    return "SAVED"

llm_engine = FunctionTool.from_defaults(
    fn=save_note,
    name ="save_notes",
    description="function to save text note file"
)