"""Evidence-linked reporting for reproducible agent evaluations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .grounding import resolve_json_pointer
from .schema import TraceValidationError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceValidationError(f"{name} must be a stable identifier")
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one local evidence input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidenceSection:
    section_id: str
    description: str
    source_sha256: Mapping[str, str]
    result: Mapping[str, Any]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier("section_id", self.section_id)
        if not isinstance(self.description, str) or not self.description.strip():
            raise TraceValidationError("section description must be non-empty text")
        if not self.source_sha256:
            raise TraceValidationError("each section requires at least one source fingerprint")
        for path, digest in self.source_sha256.items():
            if not isinstance(path, str) or not path.strip():
                raise TraceValidationError("source paths must be non-empty strings")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise TraceValidationError("source fingerprints must be lowercase SHA-256 values")
        if not isinstance(self.result, Mapping):
            raise TraceValidationError("section result must be an object")
        if not all(isinstance(item, str) and item.strip() for item in self.limitations):
            raise TraceValidationError("section limitations must be non-empty text")
        try:
            json.dumps(self.result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TraceValidationError("section result must be finite JSON data") from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceSection":
        expected = {"section_id", "description", "source_sha256", "result", "limitations"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise TraceValidationError("invalid evidence section fields")
        if not isinstance(value["limitations"], list):
            raise TraceValidationError("section limitations must be a list")
        return cls(
            value["section_id"],
            value["description"],
            value["source_sha256"],
            value["result"],
            tuple(value["limitations"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "description": self.description,
            "source_sha256": dict(self.source_sha256),
            "result": dict(self.result),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class EvidenceClaim:
    claim_id: str
    statement: str
    evidence_path: str
    expected_value: Any

    def __post_init__(self) -> None:
        _identifier("claim_id", self.claim_id)
        if not isinstance(self.statement, str) or not self.statement.strip():
            raise TraceValidationError("claim statement must be non-empty text")
        if not isinstance(self.evidence_path, str) or not self.evidence_path.startswith("/"):
            raise TraceValidationError("claim evidence_path must be a JSON pointer")
        try:
            json.dumps(self.expected_value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TraceValidationError("claim expected_value must be finite JSON data") from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceClaim":
        expected = {"claim_id", "statement", "evidence_path", "expected_value"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise TraceValidationError("invalid evidence claim fields")
        return cls(
            value["claim_id"],
            value["statement"],
            value["evidence_path"],
            value["expected_value"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "evidence_path": self.evidence_path,
            "expected_value": self.expected_value,
        }


@dataclass(frozen=True)
class EvaluationSnapshot:
    project: str
    schema_version: str
    configuration: Mapping[str, Any]
    sections: tuple[EvidenceSection, ...]
    claims: tuple[EvidenceClaim, ...]

    def __post_init__(self) -> None:
        _identifier("project", self.project)
        _identifier("schema_version", self.schema_version)
        if not self.sections:
            raise TraceValidationError("an evaluation snapshot requires evidence sections")
        for name, identifiers in (
            ("section", [item.section_id for item in self.sections]),
            ("claim", [item.claim_id for item in self.claims]),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise TraceValidationError(f"{name} identifiers must be unique")
        try:
            json.dumps(self.configuration, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TraceValidationError("snapshot configuration must be finite JSON data") from exc
        document = self._evidence_document()
        for claim in self.claims:
            observed = resolve_json_pointer(document, claim.evidence_path)
            if observed != claim.expected_value:
                raise TraceValidationError(
                    f"claim {claim.claim_id} expected value does not match its evidence"
                )

    def _evidence_document(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "schema_version": self.schema_version,
            "configuration": dict(self.configuration),
            "sections": {item.section_id: item.to_dict() for item in self.sections},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationSnapshot":
        expected = {"project", "schema_version", "configuration", "sections", "claims"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise TraceValidationError("invalid evaluation snapshot fields")
        if not isinstance(value["sections"], Mapping) or not isinstance(value["claims"], list):
            raise TraceValidationError("snapshot sections must be an object and claims a list")
        sections = tuple(EvidenceSection.from_dict(item) for item in value["sections"].values())
        return cls(
            value["project"],
            value["schema_version"],
            value["configuration"],
            sections,
            tuple(EvidenceClaim.from_dict(item) for item in value["claims"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._evidence_document(),
            "claims": [claim.to_dict() for claim in self.claims],
        }

    def render_markdown(self) -> str:
        lines = [
            "# Multimodal Agent Evaluation Report",
            "",
            "This report and its JSON snapshot were generated from the same validated",
            "in-memory evidence. All included examples are synthetic pipeline fixtures.",
            "They are not measurements of a deployed model or human review programme.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(self.configuration, indent=2, sort_keys=True),
            "```",
            "",
            "## Evidence sections",
            "",
            "| Section | Inputs | Limitations |",
            "|---|---:|---|",
        ]
        for section in self.sections:
            limitations = "; ".join(section.limitations) or "None declared"
            lines.append(f"| `{section.section_id}` | {len(section.source_sha256)} | {limitations} |")
        lines.extend(["", "## Evidence-linked claims", ""])
        for claim in self.claims:
            observed = resolve_json_pointer(self._evidence_document(), claim.evidence_path)
            lines.extend(
                [
                    f"### {claim.statement}",
                    "",
                    f"- Evidence: `{claim.evidence_path}`",
                    f"- Observed value: `{json.dumps(observed, sort_keys=True)}`",
                    "",
                ]
            )
        lines.extend(
            [
                "## Machine-readable evidence",
                "",
                "See `evaluation_snapshot.json` for source fingerprints, complete metrics,",
                "denominators, bootstrap configuration, warnings and claim bindings.",
                "",
            ]
        )
        return "\n".join(lines)
