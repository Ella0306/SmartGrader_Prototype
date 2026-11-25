import os
import json
import sqlite3
import streamlit as st
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Exam Grader - Branch Registers", layout="wide")

st.title("📝 AI-Powered Exam Grader & Branch Registers")
st.markdown("Student grades are stored securely in separate branch databases with dedicated columns for semesters and assessments.")

# --- Official List of College Branches ---
BRANCHES = [
    "Computer Science & Engineering",
    "AI/ML",
    "Electronics & Communication",
    "Mechanical",
    "Civil",
    "Electrical & Electronics"
]

# --- Helper Function: Dynamic Database Path per Branch ---
def get_db_path(branch_name):
    safe_name = "".join(c for c in branch_name if c.isalnum() or c in (' ', '_')).strip()
    return f"grades_{safe_name.lower().replace(' ', '_')}.db"

# --- STEP 1: Initialize Database with Fixed Assessment & Semester Columns ---
def init_db(branch_name):
    db_path = get_db_path(branch_name)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_grades (
            roll_number TEXT,
            name TEXT,
            branch TEXT,
            semester TEXT,
            subject_code TEXT,
            internal_1 REAL DEFAULT 0,
            internal_2 REAL DEFAULT 0,
            internal_3 REAL DEFAULT 0,
            lab REAL DEFAULT 0,
            semester_end REAL DEFAULT 0,
            PRIMARY KEY (roll_number, subject_code, semester)
        )
    """)
    conn.commit()
    conn.close()

# Helper function to convert uploaded files for Gemini
def prepare_file_part(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

class GradeResult(BaseModel):
    awarded_marks: float
    feedback: str

if not API_KEY:
    st.error("⚠️ GEMINI_API_KEY not found in your .env file!")
else:
    client = genai.Client(api_key=API_KEY)

    if "latest_eval" not in st.session_state:
        st.session_state.latest_eval = None

    # --- Sidebar: Rubric Setup & Specific Branch Database Deletion ---
    st.sidebar.header("1. Scheme of Evaluation / Rubric")
    max_marks = st.sidebar.number_input("Maximum Marks", min_value=1.0, value=10.0, step=0.5)
    
    rubric_files = st.sidebar.file_uploader(
        "Upload Model Answer / Scheme (PDF, Images)",
        type=["pdf", "png", "jpeg", "jpg"],
        accept_multiple_files=True
    )
    marking_scheme_text = st.sidebar.text_area(
        "Additional Text Instructions (Optional)",
        placeholder="E.g., Q1: 5 marks for correct definition."
    )

    # Sidebar tool to delete a specific branch database
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Database Management")
    branch_to_delete = st.sidebar.selectbox("Select Branch to Clear", BRANCHES, key="delete_branch_box")

    if st.sidebar.button(f"⚠️ Delete Branch Database", type="secondary"):
        target_db = get_db_path(branch_to_delete)
        if os.path.exists(target_db):
            os.remove(target_db)
            st.sidebar.success(f"Successfully deleted database for {branch_to_delete}!")
            st.rerun()
        else:
            st.sidebar.info(f"No database file found for {branch_to_delete}.")

    # --- Main Area: Student & Exam Metadata ---
    st.header("2. Student & Assessment Details")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        roll_number = st.text_input("Roll Number", placeholder="E.g., CSE-2026-042")
    with col2:
        student_name = st.text_input("Student Name", placeholder="E.g., John Doe")
    with col3:
        branch = st.selectbox("Branch / Department", BRANCHES)

    # Ensure table is initialized for this specific branch
    init_db(branch)

    col4, col5, col6 = st.columns(3)
    with col4:
        semester = st.selectbox("Semester", ["Semester 1", "Semester 2", "Semester 3", "Semester 4", "Semester 5", "Semester 6", "Semester 7", "Semester 8"])
    with col5:
        subject_code = st.text_input("Subject Code", placeholder="E.g., CS301")
    with col6:
        assessment_type = st.selectbox(
            "Assessment Type",
            ["Internal 1", "Internal 2", "Internal 3", "Lab Exam", "Semester End"]
        )

    st.header("3. Upload Student Answer Sheet (Multiple Pages Supported)")
    answer_files = st.file_uploader(
        "Upload Answer Sheet Pages (PDF, PNG, JPEG, JPG)",
        type=["pdf", "png", "jpeg", "jpg"],
        accept_multiple_files=True
    )

    if st.button("Evaluate Answer Sheet", type="primary"):
        if not answer_files:
            st.warning("Please upload at least one page of the student's answer sheet.")
        elif not roll_number.strip() or not student_name.strip() or not subject_code.strip():
            st.warning("Please fill in Roll Number, Student Name, and Subject Code.")
        else:
            with st.spinner(f"AI is evaluating {len(answer_files)} uploaded file(s)..."):
                try:
                    contents_payload = []
                    if rubric_files:
                        contents_payload.append("Reference Scheme of Evaluation Document(s):")
                        for r_file in rubric_files:
                            contents_payload.append(prepare_file_part(r_file))

                    contents_payload.append("Student Answer Sheet Pages:")
                    saved_files_preview = []
                    
                    for idx, a_file in enumerate(answer_files, start=1):
                        contents_payload.append(f"--- Page / File {idx}: {a_file.name} ---")
                        contents_payload.append(prepare_file_part(a_file))
                        saved_files_preview.append({
                            "bytes": a_file.getvalue(),
                            "type": a_file.type,
                            "name": a_file.name
                        })

                    prompt = f"""
                    You are an expert professor evaluating an exam paper.
                    Branch: {branch}
                    Semester: {semester}
                    Subject Code: {subject_code}
                    Assessment: {assessment_type}
                    Maximum Marks: {max_marks}
                    Instructions: {marking_scheme_text if marking_scheme_text.strip() else "Refer strictly to the attached rubric."}
                    
                    Carefully read all attached pages and return:
                    1. `awarded_marks`: Total marks earned across all pages (out of {max_marks}).
                    2. `feedback`: Question-by-question breakdown.
                    """
                    contents_payload.append(prompt)

                    # Direct single-model fast call
                    response = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=contents_payload,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=GradeResult,
                            temperature=0.2,
                        ),
                    )

                    result_data = json.loads(response.text)
                    ai_score = float(result_data.get("awarded_marks", 0))
                    ai_feedback = result_data.get("feedback", "No feedback generated.")

                    st.session_state.latest_eval = {
                        "roll_number": roll_number.strip(),
                        "name": student_name.strip(),
                        "branch": branch,
                        "semester": semester,
                        "subject_code": subject_code.strip(),
                        "assessment_type": assessment_type,
                        "ai_marks": ai_score,
                        "final_marks": ai_score,
                        "max_marks": max_marks,
                        "feedback": ai_feedback,
                        "files": saved_files_preview
                    }
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # --- Teacher Review Section ---
    if st.session_state.latest_eval is not None:
        ev = st.session_state.latest_eval
        st.divider()
        st.header("4. Teacher Review & Verification")
        
        col_prev, col_rev = st.columns(2)
        with col_prev:
            st.subheader(f"Answer Sheet Preview ({len(ev['files'])} file(s))")
            for idx, f_info in enumerate(ev["files"], start=1):
                if f_info["type"] == "application/pdf":
                    st.info(f"📄 File {idx} (PDF): {f_info['name']}")
                else:
                    st.image(f_info["bytes"], caption=f"Page {idx} - {f_info['name']}", use_container_width=True)

        with col_rev:
            st.subheader("Score Override & Confirmation")
            st.metric("AI Suggested Score", f"{ev['ai_marks']} / {ev['max_marks']}")
            
            with st.form("review_form"):
                verified_marks = st.number_input("Final Verified Marks", min_value=0.0, max_value=float(ev["max_marks"]), value=float(ev["final_marks"]), step=0.5)
                verified_feedback = st.text_area("Teacher Remarks", value=ev["feedback"], height=150)
                
                submitted = st.form_submit_button("✅ Save to Branch Database", type="primary")
                if submitted:
                    db_path = get_db_path(ev["branch"])
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    col_map = {
                        "Internal 1": "internal_1",
                        "Internal 2": "internal_2",
                        "Internal 3": "internal_3",
                        "Lab Exam": "lab",
                        "Semester End": "semester_end"
                    }
                    target_col = col_map[ev["assessment_type"]]

                    cursor.execute(f"""
                        INSERT INTO student_grades (roll_number, name, branch, semester, subject_code, {target_col})
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(roll_number, subject_code, semester) 
                        DO UPDATE SET name=excluded.name, branch=excluded.branch, {target_col}=excluded.{target_col}
                    """, (ev["roll_number"], ev["name"], ev["branch"], ev["semester"], ev["subject_code"], verified_marks))
                    
                    conn.commit()
                    conn.close()

                    st.session_state.latest_eval = None
                    st.success(f"Saved successfully to {ev['branch']} database under {ev['semester']} ({ev['assessment_type']})!")
                    st.rerun()

    # --- Display Branch Register with Semester Filter ---
    st.divider()
    st.subheader("📊 Department Marksheet Register")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        selected_view_branch = st.selectbox("Select Branch Database to View", BRANCHES, key="view_branch")
    with col_v2:
        selected_view_semester = st.selectbox("Filter by Semester", [
            "Semester 1", "Semester 2", "Semester 3", "Semester 4", 
            "Semester 5", "Semester 6", "Semester 7", "Semester 8"
        ], key="view_semester")

    db_path = get_db_path(selected_view_branch)
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("""
            SELECT roll_number, name, branch, semester, subject_code, 
                   internal_1 AS 'Internal 1', internal_2 AS 'Internal 2', 
                   internal_3 AS 'Internal 3', lab AS 'Lab', semester_end AS 'Semester End' 
            FROM student_grades 
            WHERE semester = ? 
            ORDER BY roll_number ASC
        """, conn, params=(selected_view_semester,))
        conn.close()

        if not df.empty:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(f"📥 Download {selected_view_branch} - {selected_view_semester} CSV", data=csv, file_name=f"{selected_view_branch}_{selected_view_semester}_register.csv", mime="text/csv")
        else:
            st.info(f"No grade records found for {selected_view_branch} in {selected_view_semester}.")
    else:
        st.info(f"No database file created yet for {selected_view_branch}.")