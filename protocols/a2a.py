"""A2A (Agent-to-Agent) envelope and payload skeleton.

This module defines a versioned Envelope for message passing between agents.
Extend payload models as needed; keep them in this file to satisfy the single-file app constraint.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional
from uuid import uuid4
from datetime import datetime

MsgType = Literal["task_request", "task_result", "critique", "error"]

class Envelope(BaseModel):
    schema_version: str = "1.0"
    id: str = Field(default_factory=lambda: str(uuid4()))
    ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    conversation_id: str
    from_agent: str
    to_agent: str
    intent: str
    msg_type: MsgType
    payload: Dict
    trace: Optional[Dict] = None
