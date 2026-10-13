"""
Pydantic schemas shared by every stage of the pipeline.

These are passed to Claude as structured-output schemas, so the LLM is forced
to return exactly this shape (no free-text parsing, no regex on markdown).
Every field is required; "optional" values are expressed as nullable.
"""
from typing import Literal

from pydantic import BaseModel, Field

Risk = Literal["high", "medium", "low"]
Priority = Literal["P1", "P2", "P3"]
TestType = Literal["positive", "negative", "boundary", "edge", "security", "integration"]
Technique = Literal[
    "happy_path",
    "equivalence_partitioning",
    "boundary_value",
    "decision_table",
    "state_transition",
    "error_guessing",
]
Layer = Literal["ui", "api", "db", "e2e"]
Outcome = Literal["success", "rejection", "other"]
Op = Literal["<=", "<", ">=", ">", "==", "!="]
FieldKind = Literal[
    "string", "integer", "decimal", "date", "boolean", "enum",
    "name", "email", "phone", "uuid", "account_number", "iban", "currency", "reference",
]


# ---------------------------------------------------------------- Stage 1
class Requirement(BaseModel):
    """One atomic, testable requirement extracted from release notes / specs."""
    req_id: str = Field(description="Any placeholder id - final ids are assigned by the id registry")
    module: str = Field(description="Functional area, e.g. Payments, Accounts, Auth")
    title: str
    description: str
    acceptance_criteria: list[str] = Field(description="Each criterion is one verifiable statement")
    business_rules: list[str] = Field(description="Limits, validations, formulas, thresholds")
    risk: Risk = Field(description="high = money movement, security, compliance")
    source: str = Field(description="Which input document/section it came from")


class RequirementSet(BaseModel):
    requirements: list[Requirement]


# ---------------------------------------------------------------- Stage 2
class TestStep(BaseModel):
    __test__ = False  # not a pytest class

    step: int
    action: str
    expected: str


class DataField(BaseModel):
    """A *recipe* for one data field. Python materialises the actual values,
    so the LLM never invents real customer data and volume is unlimited."""
    name: str
    kind: FieldKind
    valid: bool = Field(description="False when this field must be deliberately invalid (negative test)")
    rule: str = Field(description="Human-readable constraint, e.g. 'amount > daily limit 10000'")
    min_value: float | None
    max_value: float | None
    allowed_values: list[str] | None
    fixed_value: str | None
    reference: str | None = Field(
        description="Pull from a source dataset, e.g. 'accounts.account_id[status=ACTIVE]'")


class CrossFieldConstraint(BaseModel):
    """A rule between two values of the same data row, e.g. amount <= account_id.balance."""
    left: str = Field(description="Name of a field in data_fields")
    op: Op
    right: str = Field(description="Another field name, '<reference_field>.<column>' of the record picked "
                                   "for a reference field (e.g. 'account_id.balance'), or a literal")
    must_hold: bool = Field(description="False when the test deliberately violates this rule")


class SetupRecord(BaseModel):
    """State that must exist BEFORE the test runs, e.g. earlier transfers today."""
    entity: str = Field(description="e.g. prior_transfer, failed_login")
    description: str
    count: int
    fields: list[DataField]
    total_field: str | None = Field(description="Numeric field whose values across all records must sum to total_value")
    total_value: float | None


class TestCase(BaseModel):
    __test__ = False  # not a pytest class

    tc_id: str = Field(description="e.g. TC-REQ-PAY-001-01")
    req_id: str
    title: str
    test_type: TestType
    technique: Technique
    priority: Priority
    layer: Layer
    preconditions: list[str]
    steps: list[TestStep]
    expected_result: str
    expected_outcome: Outcome = Field(description="success, rejection, or other (e.g. OTP challenge, lockout)")
    covers_criteria: list[int] = Field(description="0-based indexes into the requirement's acceptance_criteria")
    data_fields: list[DataField]
    constraints: list[CrossFieldConstraint]
    setup: list[SetupRecord]
    automation_candidate: bool


class TestCaseSet(BaseModel):
    __test__ = False  # not a pytest class

    test_cases: list[TestCase]


# ---------------------------------------------------------------- Verification (judge)
class CaseVerdict(BaseModel):
    tc_id: str
    verified_criteria: list[int] = Field(
        description="Criteria indexes this test case GENUINELY verifies (may differ from what it claims)")
    grounded: bool = Field(description="Every expected result / value is supported by the requirement text")
    unsupported_claims: list[str] = Field(description="Statements not backed by the requirement")
    verdict: Literal["accept", "revise", "reject"]
    reason: str


class VerificationSet(BaseModel):
    verdicts: list[CaseVerdict]


# ---------------------------------------------------------------- Coverage
class CoverageRow(BaseModel):
    req_id: str
    risk: Risk
    total_criteria: int
    covered_criteria: list[int]
    uncovered_criteria: list[int]
    claimed_but_unverified: list[int]
    test_types: list[str]
    test_case_ids: list[str]
    missing_types: list[str]

    @property
    def percent(self) -> float:
        if self.total_criteria == 0:
            return 100.0
        return round(100 * len(self.covered_criteria) / self.total_criteria, 1)
