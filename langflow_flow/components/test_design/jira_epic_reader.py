"""Langflow node - read a JIRA epic and its user stories as requirements (story key = requirement id).

Alternative first node to Requirement Extractor: plug its Requirements output into Test Case Generator.
"""
# Langflow loads custom nodes by running only plain top-level imports,
# so they must not sit inside try/except. Written for Langflow >= 1.6 (lfx);
# Component is imported first to avoid a circular import inside lfx.
from lfx.custom.custom_component.component import Component
from lfx.io import BoolInput, DropdownInput, Output, SecretStrInput, StrInput
from lfx.schema.data import Data

from testgen import jira_reader
from testgen.lf_support import API_KEY_INFO, MODELS, llm_from

SOURCES = ["Local export (samples/jira)", "Live JIRA"]


class JiraEpicReaderComponent(Component):
    display_name = "JIRA Epic Reader"
    description = "Reads an epic and its user stories; each story becomes a requirement keyed by its story id."
    icon = "list-tree"
    name = "TestgenJiraEpicReader"

    inputs = [
        StrInput(name="epic_key", display_name="Epic key", value="PAY-100", info="e.g. PAY-100"),
        DropdownInput(name="source", display_name="Read from", options=SOURCES, value=SOURCES[0],
                      info="Local export = samples/jira/<EPIC>.json (demo, no JIRA access needed)"),
        StrInput(name="local_folder", display_name="Local export folder", value="samples/jira", advanced=True),
        StrInput(name="jira_url", display_name="JIRA URL", value="", required=False,
                 info="e.g. https://yourcompany.atlassian.net - empty = JIRA_URL from .env"),
        StrInput(name="jira_user", display_name="JIRA email (Cloud only)", value="", required=False,
                 info="Cloud: your Atlassian email. Server/Data Center: leave empty and use a personal access token"),
        SecretStrInput(name="jira_token", display_name="JIRA API token / PAT", required=False,
                       info="Empty = JIRA_API_TOKEN or JIRA_PAT from .env"),
        StrInput(name="ac_field", display_name="Acceptance criteria field id", value="", advanced=True,
                 info="e.g. customfield_10045. Empty = auto-detect a field named 'Acceptance Criteria', "
                      "else the 'Acceptance Criteria' section of the description"),
        # The LLM is only used for stories that have NO acceptance criteria at all
        SecretStrInput(name="api_key", display_name="API Key (Claude or Gemini)", required=False, info=API_KEY_INFO,
                       advanced=True),
        DropdownInput(name="model", display_name="Model (stories without criteria)", options=MODELS,
                      value="gemini-3.6-flash", combobox=True, advanced=True),
        BoolInput(name="offline", display_name="Offline (no LLM)", value=False, advanced=True),
    ]
    outputs = [Output(display_name="Requirements", name="requirements", method="read")]

    def read(self) -> list[Data]:
        live = self.source == SOURCES[1]
        source = jira_reader.get_source("jira" if live else "file", self.local_folder,
                                        self.jira_url, self.jira_user, self.jira_token)
        llm = llm_from(self)
        reqs, info = jira_reader.read_epic(self.epic_key.strip(), source, self.ac_field.strip() or None, llm=llm)
        if not reqs:
            raise ValueError(f"Epic {self.epic_key} has no stories with a description or acceptance criteria")
        extra = []
        if info["llm_extracted"]:
            extra.append(f"criteria extracted by {llm.name} for {info['llm_extracted']}")
        if info["skipped_no_content"]:
            extra.append(f"skipped empty {info['skipped_no_content']}")
        self.status = (f"{info['epic']} '{info['epic_summary']}': {len(info['stories'])} stories -> "
                       f"{len(reqs)} requirements" + (" | " + "; ".join(extra) if extra else ""))
        return [Data(data=r.model_dump()) for r in reqs]
