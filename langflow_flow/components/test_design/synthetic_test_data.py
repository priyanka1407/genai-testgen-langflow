"""Langflow node 3/4 - data recipes -> runner-ready synthetic test data."""
from pathlib import Path

# Langflow loads custom nodes by running only plain top-level imports,
# so they must not sit inside try/except. Written for Langflow >= 1.6 (lfx);
# Component is imported first to avoid a circular import inside lfx.
from lfx.custom.custom_component.component import Component
from lfx.io import DataInput, IntInput, Output, StrInput
from lfx.schema.data import Data

from testgen import exporters, ingest, testdata
from testgen.lf_support import as_dicts
from testgen.models import TestCase


class SyntheticTestDataComponent(Component):
    display_name = "Synthetic Test Data"
    description = "Consolidates source datasets and materialises each test case's data recipe as runner-ready JSON."
    icon = "database"
    name = "TestgenSyntheticData"

    inputs = [
        DataInput(name="test_cases", display_name="Test Cases", is_list=True),
        StrInput(name="data_path", display_name="Data sources folder", value="samples/data"),
        IntInput(name="rows_per_case", display_name="Rows per test case", value=3),
        StrInput(name="out_dir", display_name="Output folder", value="output"),
    ]
    outputs = [Output(display_name="Test Data", name="test_data", method="build")]

    def build(self) -> list[Data]:
        cases = [TestCase(**d) for d in as_dicts(self.test_cases)]
        sources, _ = ingest.consolidate(ingest.load_data_sources([self.data_path]) if self.data_path else {})
        dataset = testdata.build_dataset(cases, sources, self.rows_per_case)
        n = exporters.write_testdata(dataset, cases, Path(self.out_dir))
        self.status = f"{n} rows -> {self.out_dir}/testdata"
        return [Data(data={"tc_id": k, "rows": v}) for k, v in dataset.items() if v]
