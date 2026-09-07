"""
ai_engine.py
------------
Core AI workflow engine for the Study Pack Generator.

Pipeline (5 stages, each reads/writes a shared WorkflowContext):
  1. Planning            topic+goals        -> structured study plan
  2. Content Generation   plan               -> lesson content per topic
  3. Assessment           content            -> quizzes per topic
  4. Review               content+assessment -> quality critique
  5. Refinement           review flags       -> revised content/assessment

Design notes:
- WorkflowContext is the single object passed stage-to-stage ("context passing").
- LLMClient wraps every model call with retry/backoff + JSON auto-repair.
- StudyPackWorkflow.run() executes stages in order, catches per-stage errors,
  logs them into the context, and returns the context however far it got
  (partial results) instead of throwing away everything on one failure.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("study_pack_engine")

MODEL_NAME = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# 1. Shared context object — this is what "context passing" means here
# ---------------------------------------------------------------------------
@dataclass
class WorkflowContext:
    topic: str
    level: str            # "beginner" | "intermediate" | "advanced"
    duration_days: int
    goals: str

    plan: Optional[dict] = None
    content: Optional[dict] = None
    assessment: Optional[dict] = None
    review: Optional[dict] = None
    final_pack: Optional[dict] = None

    stage_log: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def log_stage(self, name: str, status: str, detail: str = ""):
        self.stage_log.append({"stage": name, "status": status, "detail": detail})

    def log_error(self, stage: str, error: str):
        self.errors.append({"stage": stage, "error": error})
        logger.error(f"[{stage}] {error}")

    @property
    def failed(self) -> bool:
        return any(s["status"] == "failed" for s in self.stage_log)


# ---------------------------------------------------------------------------
# 2. LLM client — retries + JSON repair live here, not scattered in stages
# ---------------------------------------------------------------------------
class LLMClient:
    def __init__(self, api_key: str, model: str = MODEL_NAME, max_retries: int = 3):
        if not api_key:
            raise ValueError("API key is required to create an LLMClient.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def call(self, system: str, prompt: str, max_tokens: int = 2500) -> str:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    f"LLM call failed (attempt {attempt}/{self.max_retries}): {e}. "
                    f"Retrying in {wait}s"
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def call_json(self, system: str, prompt: str, max_tokens: int = 2500) -> dict:
        """Call the model and parse JSON, with a single repair pass on failure."""
        raw = self.call(system, prompt, max_tokens)
        try:
            return self._extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Primary JSON parse failed — attempting repair pass")
            repair_prompt = (
                "The text below was supposed to be valid JSON but does not parse. "
                "Return ONLY the corrected, valid JSON object — no commentary, "
                "no markdown fences:\n\n" + raw
            )
            repaired = self.call(system, repair_prompt, max_tokens)
            return self._extract_json(repaired)  # let it raise if still broken

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.split("json", 1)[-1] if text.lower().startswith("json") else text
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model response.")
        return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# 3. The workflow itself
# ---------------------------------------------------------------------------
class StudyPackWorkflow:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ---- Stage 1: Planning -------------------------------------------------
    def plan_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        system = (
            "You are an expert curriculum designer. Output ONLY valid JSON, "
            "no prose, no markdown fences."
        )
        prompt = f"""
Create a study plan as JSON with this exact shape:
{{
  "topics": [
    {{"name": "...", "objectives": ["..."], "est_minutes": 30, "order": 1}}
  ],
  "summary": "one paragraph overview of the plan"
}}

Subject: {ctx.topic}
Learner level: {ctx.level}
Available time: {ctx.duration_days} day(s)
Learner goals: {ctx.goals}

Break the subject into 4-8 topics sized to fit the available time.
"""
        try:
            ctx.plan = self.llm.call_json(system, prompt)
            ctx.log_stage("planning", "success", f"{len(ctx.plan.get('topics', []))} topics")
        except Exception as e:
            ctx.log_error("planning", str(e))
            ctx.log_stage("planning", "failed", str(e))
        return ctx

    # ---- Stage 2: Content Generation ---------------------------------------
    def content_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        if not ctx.plan:
            ctx.log_stage("content_generation", "skipped", "no plan available")
            return ctx
        system = (
            "You are a patient, clear subject-matter tutor. Output ONLY valid JSON."
        )
        prompt = f"""
Using this study plan (context from the previous stage), write lesson content
for EVERY topic. Match the learner level: {ctx.level}.

Plan:
{json.dumps(ctx.plan, indent=2)}

Return JSON exactly shaped as:
{{
  "lessons": [
    {{
      "topic": "must match a topic name from the plan",
      "explanation": "clear explanation, 150-300 words",
      "key_points": ["...", "..."],
      "example": "one worked example or analogy"
    }}
  ]
}}
"""
        try:
            ctx.content = self.llm.call_json(system, prompt, max_tokens=4000)
            ctx.log_stage("content_generation", "success", f"{len(ctx.content.get('lessons', []))} lessons")
        except Exception as e:
            ctx.log_error("content_generation", str(e))
            ctx.log_stage("content_generation", "failed", str(e))
        return ctx

    # ---- Stage 3: Assessment ------------------------------------------------
    def assessment_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        if not ctx.content:
            ctx.log_stage("assessment", "skipped", "no content available")
            return ctx
        system = "You are an assessment writer. Output ONLY valid JSON."
        prompt = f"""
Using this lesson content (context from the previous stage), write a short quiz
for EACH topic: 3 multiple-choice questions with 4 options, the correct answer,
and a one-line explanation.

Content:
{json.dumps(ctx.content, indent=2)}

Return JSON exactly shaped as:
{{
  "quizzes": [
    {{
      "topic": "must match a topic name",
      "questions": [
        {{
          "question": "...",
          "options": ["A...", "B...", "C...", "D..."],
          "correct_answer": "A...",
          "explanation": "..."
        }}
      ]
    }}
  ]
}}
"""
        try:
            ctx.assessment = self.llm.call_json(system, prompt, max_tokens=4000)
            ctx.log_stage("assessment", "success", f"{len(ctx.assessment.get('quizzes', []))} quizzes")
        except Exception as e:
            ctx.log_error("assessment", str(e))
            ctx.log_stage("assessment", "failed", str(e))
        return ctx

    # ---- Stage 4: Review ------------------------------------------------------
    def review_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        if not ctx.content or not ctx.assessment:
            ctx.log_stage("review", "skipped", "missing content or assessment")
            return ctx
        system = (
            "You are a strict quality reviewer for educational material. "
            "Output ONLY valid JSON."
        )
        prompt = f"""
Review this lesson content and quiz set against the original plan for accuracy,
clarity, and alignment with stated goals: {ctx.goals}

Plan: {json.dumps(ctx.plan, indent=2)}
Content: {json.dumps(ctx.content, indent=2)}
Assessment: {json.dumps(ctx.assessment, indent=2)}

Return JSON exactly shaped as:
{{
  "overall_quality_score": 0-10,
  "issues": [
    {{"topic": "...", "severity": "low|medium|high", "problem": "...", "suggestion": "..."}}
  ],
  "needs_refinement": true/false
}}
"""
        try:
            ctx.review = self.llm.call_json(system, prompt, max_tokens=2000)
            ctx.log_stage(
                "review", "success",
                f"score={ctx.review.get('overall_quality_score')} "
                f"issues={len(ctx.review.get('issues', []))}"
            )
        except Exception as e:
            ctx.log_error("review", str(e))
            ctx.log_stage("review", "failed", str(e))
        return ctx

    # ---- Stage 5: Refinement --------------------------------------------------
    def refinement_stage(self, ctx: WorkflowContext) -> WorkflowContext:
        if not ctx.review or not ctx.review.get("needs_refinement"):
            # nothing flagged, or review stage failed — pass content through as-is
            ctx.final_pack = {"content": ctx.content, "assessment": ctx.assessment, "plan": ctx.plan}
            ctx.log_stage("refinement", "skipped", "no refinement needed or review unavailable")
            return ctx

        system = "You are an editor fixing flagged issues in educational content. Output ONLY valid JSON."
        prompt = f"""
Revise the lesson content and quizzes below to fix ONLY the listed issues.
Keep everything else unchanged. Preserve the same JSON shapes.

Issues to fix:
{json.dumps(ctx.review.get('issues', []), indent=2)}

Current content:
{json.dumps(ctx.content, indent=2)}

Current assessment:
{json.dumps(ctx.assessment, indent=2)}

Return JSON exactly shaped as:
{{
  "lessons": [ ... same shape as before ... ],
  "quizzes": [ ... same shape as before ... ]
}}
"""
        try:
            revised = self.llm.call_json(system, prompt, max_tokens=4000)
            ctx.content = {"lessons": revised.get("lessons", ctx.content.get("lessons", []))}
            ctx.assessment = {"quizzes": revised.get("quizzes", ctx.assessment.get("quizzes", []))}
            ctx.final_pack = {"content": ctx.content, "assessment": ctx.assessment, "plan": ctx.plan}
            ctx.log_stage("refinement", "success", "issues addressed")
        except Exception as e:
            # fall back to unrefined but usable content rather than losing everything
            ctx.log_error("refinement", str(e))
            ctx.final_pack = {"content": ctx.content, "assessment": ctx.assessment, "plan": ctx.plan}
            ctx.log_stage("refinement", "failed", f"{e} — falling back to unrefined content")
        return ctx

    # ---- Orchestrator -----------------------------------------------------
    def run(self, ctx: WorkflowContext, progress_callback=None) -> WorkflowContext:
        """
        Runs all 5 stages in order. Stops early only if a stage that later
        stages strictly depend on produced nothing usable (e.g. planning fails).
        Always returns the context with whatever was completed.
        """
        stages = [
            ("planning", self.plan_stage),
            ("content_generation", self.content_stage),
            ("assessment", self.assessment_stage),
            ("review", self.review_stage),
            ("refinement", self.refinement_stage),
        ]
        for name, fn in stages:
            if progress_callback:
                progress_callback(name, "running")
            ctx = fn(ctx)
            if progress_callback:
                last = ctx.stage_log[-1]["status"] if ctx.stage_log else "unknown"
                progress_callback(name, last)
            # hard stop only if the foundational planning stage failed
            if name == "planning" and ctx.plan is None:
                break
        return ctx
