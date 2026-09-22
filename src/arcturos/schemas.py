"""Pydantic request schemas for the Arcturos API.

All create payloads are validated here before touching the database.
Timestamps are owned by the server (UTC, ISO-8601) so stored rows keep an
honest append-only audit trail.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RunCreate(BaseModel):
    server_url: str = Field(..., pattern=r"^https?://\S+$")
    model_fingerprint: str = Field(..., min_length=1)
    engine: str = Field(..., min_length=1)
    context_size: int = Field(..., ge=1)
    created_at: str = Field(default_factory=_utcnow)


class BenchmarkCreate(BaseModel):
    context_tokens: int = Field(..., ge=1)
    prefill_tps: Optional[float] = Field(None, gt=0)
    decode_tps: Optional[float] = Field(None, gt=0)
    ttft_ms: Optional[float] = Field(None, ge=0)
    wall_s: Optional[float] = Field(None, gt=0)
    output_tokens: Optional[int] = Field(None, ge=0)
    mtp_draft_n: Optional[int] = Field(None, ge=0)
    mtp_accepted: Optional[int] = Field(None, ge=0)
    power_watts: Optional[float] = Field(None, ge=0)
    power_host: Optional[str] = Field(None, min_length=1)
    power_gpu_index: Optional[int] = Field(None, ge=0)
    streams: Optional[int] = Field(1, ge=1, le=16)
    decode_tps_combined: Optional[float] = Field(None, gt=0)
    prefill_tps_combined: Optional[float] = Field(None, gt=0)
    created_at: str = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_power_consistency(self) -> "BenchmarkCreate":
        # A power number without provenance is unverifiable — reject it so
        # no stored row can ever show watts with a missing host (ADR 002).
        if self.power_watts is not None and not self.power_host:
            raise ValueError("power_watts requires power_host (provenance)")
        if self.power_host is not None and self.power_watts is None:
            raise ValueError("power_host without power_watts is meaningless")
        return self

    @model_validator(mode="after")
    def _check_mtp_acceptance(self) -> "BenchmarkCreate":
        if self.mtp_draft_n is not None and self.mtp_accepted is not None:
            if self.mtp_accepted > self.mtp_draft_n:
                raise ValueError("mtp_accepted cannot exceed mtp_draft_n")
        return self


class ModelAliasUpdate(BaseModel):
    """Only the alias is editable — the fingerprint stays immutable (§4.6)."""
    alias: Optional[str] = Field(None, max_length=120)


class BaselineCreate(BaseModel):
    model_fingerprint: str = Field(..., min_length=1)
    # allowlist keeps metric families honest across views
    metric_family: Literal["speed", "quality"]
    run_id: int = Field(..., ge=1)
    created_at: str = Field(default_factory=_utcnow)


class EvalSuiteCreate(BaseModel):
    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


class EvalResultCreate(BaseModel):
    model_fingerprint: str = Field(..., min_length=1)
    item_id: str = Field(..., min_length=1)
    output: str
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    created_at: str = Field(default_factory=_utcnow)


class JudgmentCreate(BaseModel):
    eval_result_a: int = Field(..., ge=1)
    eval_result_b: int = Field(..., ge=1)
    judge_model: str = Field(..., min_length=1)
    judge_template_version: str = Field(..., min_length=1)
    winner: Literal["a", "b", "tie"]
    confidence: Optional[float] = Field(None, ge=0, le=1)
    rationale: Optional[str] = None
    created_at: str = Field(default_factory=_utcnow)

    @model_validator(mode="after")
    def _check_distinct_results(self) -> "JudgmentCreate":
        if self.eval_result_a == self.eval_result_b:
            raise ValueError("eval_result_a and eval_result_b must differ")
        return self
