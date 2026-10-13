"""Learning step 3: the smallest possible custom node - one input, one output."""
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message


class HelloComponent(Component):
    display_name = "Hello"                      # title on the node
    description = "My first custom node."       # text under the title
    icon = "hand"                               # any Lucide icon name
    name = "LearningHello"                      # unique internal id

    # Each input becomes a field on the node; read it as self.<name>
    inputs = [MessageTextInput(name="who", display_name="Name", value="world")]

    # Each output becomes a dot on the right; Langflow calls `method` to produce it
    outputs = [Output(display_name="Greeting", name="greeting", method="greet")]

    def greet(self) -> Message:                 # Message = blue dot (text), e.g. into Chat Output
        self.status = "greeted"                 # text shown on the node after it runs
        return Message(text=f"Hello {self.who}!")
