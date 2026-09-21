"""Rendered prompt compatible with existing string consumers and snapshot JSON."""

class RuntimePrompt(str):
    def __new__(cls, stable_instructions: str, dynamic_context: str = ""):
        value = super().__new__(cls, "\n\n".join(filter(None, (stable_instructions, dynamic_context))))
        value.stable_instructions = stable_instructions
        value.dynamic_context = dynamic_context
        return value
