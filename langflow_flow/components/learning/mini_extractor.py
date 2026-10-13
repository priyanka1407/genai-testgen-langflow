"""Learning step 4: a minimal requirement extractor - release notes in, structured requirements out.

This is the core idea of the real Requirement Extractor in ~50 lines, calling Gemini directly.
"""
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, MultilineInput, Output, SecretStrInput
from lfx.schema.data import Data

from google import genai
from google.genai import types
from pydantic import BaseModel


# 1. The SHAPE of the answer we want. Gemini is forced to return exactly this.
class Requirement(BaseModel):
    title: str
    acceptance_criteria: list[str]
    risk: str            # "high" | "medium" | "low"


class RequirementList(BaseModel):
    requirements: list[Requirement]


# 2. The INSTRUCTIONS. This is where the "understanding" of release notes is steered.
PROMPT = """You are a QA analyst. Split the release notes below into small, testable requirements.
- One requirement = one behaviour that can pass or fail on its own.
- acceptance_criteria: short, verifiable statements. Copy every number and limit exactly.
- risk: high for money movement, security or compliance; otherwise medium or low.

Release notes:
{notes}"""


class MiniRequirementExtractor(Component):
    display_name = "Mini Requirement Extractor"
    description = "Practice node: release notes -> structured requirements with Gemini."
    icon = "file-search"
    name = "LearningMiniExtractor"

    inputs = [
        MultilineInput(name="notes", display_name="Release notes", info="Paste your release notes here"),
        SecretStrInput(name="api_key", display_name="Gemini API Key", required=False,
                       info="Leave empty to use GEMINI_API_KEY from .env"),
        DropdownInput(name="model", display_name="Model", combobox=True,
                      options=["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"], value="gemini-3.6-flash"),
    ]
    outputs = [Output(display_name="Requirements", name="requirements", method="extract")]

    def extract(self) -> list[Data]:
        # 3. CALL the model with the prompt and the required answer shape
        client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        response = client.models.generate_content(
            model=self.model,
            contents=PROMPT.format(notes=self.notes),
            config=types.GenerateContentConfig(response_mime_type="application/json",
                                               response_schema=RequirementList),
        )
        # 4. VALIDATE the answer and pass it on as Data (red dot) to the next node
        result = response.parsed or RequirementList.model_validate_json(response.text)
        self.status = f"{len(result.requirements)} requirements"
        return [Data(data=r.model_dump()) for r in result.requirements]
