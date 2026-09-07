"""
utils.py
--------
Helper functions used by app.py: input validation and turning the
final WorkflowContext into a downloadable Markdown study pack.
"""

from datetime import datetime


def validate_inputs(topic: str, duration_days: int, goals: str) -> list:
    """Returns a list of error strings; empty list means inputs are OK."""
    errors = []
    if not topic or not topic.strip():
        errors.append("Please enter a topic/subject.")
    if duration_days is None or duration_days < 1:
        errors.append("Duration must be at least 1 day.")
    if duration_days is not None and duration_days > 60:
        errors.append("Duration over 60 days is probably a typo — please double check.")
    if not goals or not goals.strip():
        errors.append("Please describe your learning goal (even one sentence helps).")
    return errors


def context_to_markdown(ctx) -> str:
    """Render a WorkflowContext into a single Markdown document for download."""
    lines = []
    lines.append(f"# Study Pack: {ctx.topic}")
    lines.append(f"*Level: {ctx.level} | Duration: {ctx.duration_days} day(s) | Generated: "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"\n**Goal:** {ctx.goals}\n")

    if ctx.plan and ctx.plan.get("summary"):
        lines.append("## Overview")
        lines.append(ctx.plan["summary"])

    lessons = (ctx.final_pack or {}).get("content", {}).get("lessons", [])
    quizzes = (ctx.final_pack or {}).get("assessment", {}).get("quizzes", [])
    quiz_by_topic = {q["topic"]: q for q in quizzes} if quizzes else {}

    for lesson in lessons:
        topic = lesson.get("topic", "Untitled topic")
        lines.append(f"\n## {topic}")
        lines.append(lesson.get("explanation", ""))

        if lesson.get("key_points"):
            lines.append("\n**Key points:**")
            for kp in lesson["key_points"]:
                lines.append(f"- {kp}")

        if lesson.get("example"):
            lines.append(f"\n**Example:** {lesson['example']}")

        quiz = quiz_by_topic.get(topic)
        if quiz and quiz.get("questions"):
            lines.append("\n**Quick check:**")
            for i, q in enumerate(quiz["questions"], 1):
                lines.append(f"\n{i}. {q['question']}")
                for opt in q.get("options", []):
                    lines.append(f"   - {opt}")
                lines.append(f"   - *Answer: {q.get('correct_answer', '')} — {q.get('explanation', '')}*")

    if ctx.review:
        lines.append(f"\n---\n*Quality score: {ctx.review.get('overall_quality_score', 'N/A')}/10*")

    if ctx.errors:
        lines.append("\n---\n### Notes")
        lines.append("Some stages had issues and were skipped or fell back to earlier results:")
        for e in ctx.errors:
            lines.append(f"- **{e['stage']}**: {e['error']}")

    return "\n".join(lines)


def stage_progress_summary(ctx) -> list:
    """Compact list of (stage, status) for a UI progress display."""
    return [(s["stage"], s["status"]) for s in ctx.stage_log]
