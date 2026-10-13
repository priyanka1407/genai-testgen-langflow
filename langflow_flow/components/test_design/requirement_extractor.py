"""Langflow node 1/4 - release notes & specs -> atomic testable requirements (stable ids)."""
# Langflow loads custom nodes by running only plain top-level imports,
# so they must not sit inside try/except. Written for Langflow >= 1.6 (lfx);
# Component is imported first to avoid a circular import inside lfx.
from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DropdownInput, MultilineInput, Output, SecretStrInput, StrInput
from lfx.schema.data import Data

from testgen import generator, ingest
from testgen.lf_support import API_KEY_INFO, MODELS, llm_from
from testgen.state import State


class RequirementExtractorComponent(Component):
    display_name = "Requirement Extractor"
    description = "Turns release notes + requirement specs into atomic, testable requirements."
    icon = "file-search"
    name = "TestgenRequirementExtractor"

    inputs = [
        StrInput(name="docs_path", display_name="Docs folder / file", value="samples/docs",
                 info="Folder with .md/.txt/.docx/.pdf release notes and specs"),
        MultilineInput(name="pasted_text", display_name="Or paste release notes", value=""),
        StrInput(name="state_dir", display_name="State folder", value="state",
                 info="Shared with the CLI pipeline - keeps requirement ids stable across releases"),
        SecretStrInput(name="api_key", display_name="API Key (Claude or Gemini)", required=False, info=API_KEY_INFO),
        DropdownInput(name="model", display_name="Model", options=MODELS, value=MODELS[0], combobox=True,
                      info="gemini-* uses Gemini, claude-* uses Claude. You can also type a model id."),
        BoolInput(name="offline", display_name="Offline (no LLM)", value=False, advanced=True),
    ]
    outputs = [Output(display_name="Requirements", name="requirements", method="extract")]

    def extract(self) -> list[Data]:
        docs = ingest.load_documents([self.docs_path]) if self.docs_path else []
        if self.pasted_text and self.pasted_text.strip():
            docs.append(("pasted_input.md", self.pasted_text))
        st = State.load(self.state_dir)
        llm = llm_from(self)
        reqs = generator.extract_requirements(llm, docs, previous=st.previous_requirements())
        statuses, removed = st.assign_ids(reqs)  # stable REQ ids; change detection happens in Test Case Generator
        st.save()                                # remember the id registry for the next run
        counts = {k: sum(1 for v in statuses.values() if v == k) for k in ("new", "changed", "unchanged")}
        self.status = f"{len(reqs)} requirements {counts}, removed {removed or 'none'} | model: {llm.name}"
        return [Data(data=r.model_dump()) for r in reqs]
