"""
app.py
------
Streamlit front-end for the AI Study Pack Generator.
Run locally with:  streamlit run app.py
"""

import streamlit as st
from ai_engine import WorkflowContext, LLMClient, StudyPackWorkflow
from utils import validate_inputs, context_to_markdown

st.set_page_config(page_title="AI Study Pack Generator", page_icon="📚", layout="wide")

st.title("📚 AI Study Pack Generator")
st.caption("Planning → Content → Assessment → Review → Refinement, powered by a 5-stage AI workflow.")

# ---------------------------------------------------------------------------
# Sidebar: API key + inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Anthropic API key", type="password",
                             help="Get one at console.anthropic.com. Not stored anywhere.")
    st.divider()
    st.header("Study Pack Details")
    topic = st.text_input("Topic / Subject", placeholder="e.g. Photosynthesis")
    level = st.selectbox("Learner level", ["beginner", "intermediate", "advanced"])
    duration_days = st.number_input("Duration (days)", min_value=1, max_value=60, value=3)
    goals = st.text_area("Learning goal", placeholder="e.g. Pass my Grade 10 biology exam")
    generate_clicked = st.button("Generate Study Pack", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "ctx" not in st.session_state:
    st.session_state.ctx = None

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------
if generate_clicked:
    errors = validate_inputs(topic, duration_days, goals)
    if not api_key:
        errors.append("Please enter your Anthropic API key in the sidebar.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        ctx = WorkflowContext(topic=topic, level=level, duration_days=duration_days, goals=goals)

        status_box = st.status("Running AI workflow...", expanded=True)
        stage_labels = {
            "planning": "1️⃣ Planning",
            "content_generation": "2️⃣ Content generation",
            "assessment": "3️⃣ Assessment",
            "review": "4️⃣ Review",
            "refinement": "5️⃣ Refinement",
        }

        def progress_callback(stage_name, status):
            label = stage_labels.get(stage_name, stage_name)
            if status == "running":
                status_box.write(f"{label}: running...")
            elif status == "success":
                status_box.write(f"{label}: ✅ done")
            elif status == "skipped":
                status_box.write(f"{label}: ⏭️ skipped")
            elif status == "failed":
                status_box.write(f"{label}: ⚠️ failed (will try to continue)")

        try:
            llm = LLMClient(api_key=api_key)
            workflow = StudyPackWorkflow(llm)
            ctx = workflow.run(ctx, progress_callback=progress_callback)
            st.session_state.ctx = ctx

            if ctx.plan is None:
                status_box.update(label="Workflow failed at planning stage", state="error")
            elif ctx.errors:
                status_box.update(label="Workflow finished with some issues", state="complete")
            else:
                status_box.update(label="Workflow complete", state="complete")
        except Exception as e:
            status_box.update(label="Workflow crashed", state="error")
            st.error(f"Could not run the workflow: {e}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
ctx = st.session_state.ctx
if ctx and ctx.final_pack:
    st.divider()
    st.header(f"Your Study Pack: {ctx.topic}")

    if ctx.review:
        col1, col2 = st.columns(2)
        col1.metric("Quality score", f"{ctx.review.get('overall_quality_score', 'N/A')}/10")
        col2.metric("Issues found & handled", len(ctx.review.get("issues", [])))

    lessons = ctx.final_pack.get("content", {}).get("lessons", [])
    quizzes = ctx.final_pack.get("assessment", {}).get("quizzes", [])
    quiz_by_topic = {q["topic"]: q for q in quizzes}

    for lesson in lessons:
        with st.expander(f"📖 {lesson.get('topic', 'Topic')}", expanded=False):
            st.write(lesson.get("explanation", ""))
            if lesson.get("key_points"):
                st.markdown("**Key points:**")
                for kp in lesson["key_points"]:
                    st.markdown(f"- {kp}")
            if lesson.get("example"):
                st.markdown(f"**Example:** {lesson['example']}")

            quiz = quiz_by_topic.get(lesson.get("topic"))
            if quiz and quiz.get("questions"):
                st.markdown("**Quick check:**")
                for i, q in enumerate(quiz["questions"], 1):
                    st.markdown(f"{i}. {q['question']}")
                    choice = st.radio(
                        "Choose:", q.get("options", []),
                        key=f"{lesson.get('topic')}_{i}", label_visibility="collapsed"
                    )
                    if st.button("Check answer", key=f"check_{lesson.get('topic')}_{i}"):
                        if choice == q.get("correct_answer"):
                            st.success(f"Correct! {q.get('explanation', '')}")
                        else:
                            st.warning(f"Not quite. Correct answer: {q.get('correct_answer')} — {q.get('explanation', '')}")

    if ctx.errors:
        with st.expander("⚠️ Workflow notes"):
            for e in ctx.errors:
                st.write(f"**{e['stage']}**: {e['error']}")

    st.divider()
    md = context_to_markdown(ctx)
    st.download_button(
        "⬇️ Download study pack (Markdown)",
        data=md,
        file_name=f"study_pack_{ctx.topic.replace(' ', '_')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
elif ctx and ctx.plan is None:
    st.warning("Planning stage failed — try again, or check your API key/quota.")
else:
    st.info("Fill in the sidebar and click **Generate Study Pack** to start.")
