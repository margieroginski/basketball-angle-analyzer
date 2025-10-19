import os
import sys
import time
import platform
import subprocess
import streamlit as st

st.set_page_config(page_title="Basketball Angle Analyzer", layout="centered")

st.title("🏀 Basketball Angle Analyzer")
st.caption(f"Python: {sys.version.split()[0]} • Platform: {platform.platform()}")

st.write("Upload a side-view basketball video and choose which body angles to annotate.")

# ────────────────────────────────────────────────────────────────────────────────
# File upload
# ────────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your .mp4 video", type=["mp4"])
input_path = "input.mp4"
output_path = "annotated.mp4"

if uploaded_file:
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ Uploaded: {uploaded_file.name}")

# ────────────────────────────────────────────────────────────────────────────────
# Options (match annotate_angles.py)
# ────────────────────────────────────────────────────────────────────────────────

# Angles with descriptions (for helper text)
ANGLE_OPTIONS = {
    "shin": "Shin angle vs vertical (draws ankle vertical reference)",
    "knee": "Knee flexion (0 = straight, higher = more bent)",
    "knee_pos": "Knee position vs toe tip (Behind / Over / In front)",
    "elbow": "Elbow flexion (0 = straight, higher = more bent)",
    "elbow_height": "Elbow vertical offset vs shoulder (pixels; + above / - below)",
    "hip": "Hip flexion (0 = upright, higher = more flexed)",
    "torso": "Torso lean vs vertical (0 = vertical)",
    "head": "Head tilt vs vertical (shoulder anchor)",
    "hand": "Wrist/hand flexion (0 = straight, higher = more bent)",
}

st.subheader("Angles")
with st.expander("Select angles to display", expanded=True):
    default_angles = ["shin", "knee", "knee_pos", "hip", "torso", "elbow", "hand"]
    chosen_angles = st.multiselect(
        "Angles",
        options=list(ANGLE_OPTIONS.keys()),
        default=default_angles,
        help="\n".join([f"{k}: {v}" for k, v in ANGLE_OPTIONS.items()]),
    )

st.subheader("Overlay & Skeleton")
cols = st.columns(2)
with cols[0]:
    show_overlay = st.checkbox("Show overlay panel", True)
    overlay_position = st.selectbox(
        "Overlay position",
        ["top-right", "top-left", "bottom-right", "bottom-left"],
        index=0,
    )
with cols[1]:
    show_skeleton = st.checkbox("Show skeleton", False)
#    smoothing = st.slider("Smoothing (`--smoothing`)", min_value=1, max_value=15, value=5)

st.subheader("Appearance (sizes)")
c1, c2 = st.columns(2)
with c1:
    font_scale = st.slider("Font scale", 0.1, 3.0, 0.5, 0.1)
    text_thickness = st.slider("Text thickness", 1, 6, 1)
with c2:
    line_thickness = st.slider("Line thickness", 1, 10, 2)
    dot_radius = st.slider("Joint dot radius", 1, 12, 3)


# Detect Streamlit Cloud-ish environment
is_cloud = os.environ.get("STREAMLIT_RUNTIME", "") == "cloud" or "streamlit" in platform.platform().lower()

# ────────────────────────────────────────────────────────────────────────────────
# Run
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
run_clicked = st.button("Process Video")

if run_clicked:
    if not uploaded_file:
        st.warning("Please upload a video first.")
    else:
        # Build CLI command
        cmd = [
            sys.executable,  # use current interpreter
            "annotate_angles.py",
            "--input", input_path,
            "--output", output_path,
            "--angles", ",".join(chosen_angles) if chosen_angles else "",
#            "--smoothing", str(smoothing),
            "--overlay-position", overlay_position,
            "--font-scale", str(font_scale),
            "--text-thickness", str(text_thickness),
            "--line-thickness", str(line_thickness),
            "--dot-radius", str(dot_radius),
        ]
        if show_overlay:
            cmd.append("--show-overlay")
        if show_skeleton:
            cmd.append("--show-skeleton")

        st.write("🚀 Running analysis... This may take a minute.")
        st.write("```" + " ".join(cmd) + "```")

        import time

        with st.spinner("Processing video..."):
            subprocess.run(cmd)

        # Wait briefly to ensure OpenCV flushes and file is fully written
        time.sleep(1)

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            st.error("❌ Output video appears empty — processing may have failed.")
        else:
            st.success("✅ Processing complete!")
            st.video(output_path)
#            with open(output_path, "rb") as f:
#                video_bytes = f.read()
#            st.video(video_bytes)
            
            #st.video(output_path)
            print(f"opening {output_path}")
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download annotated video",
                    data=f,
                    file_name="annotated.mp4",
                    mime="video/mp4"
                )

#         st.write("🚀 Running:")
#         st.code(" ".join([str(c) for c in cmd]), language="bash")
# 
#         with st.spinner("Processing video..."):
#             # Run annotate script
#             result = subprocess.run(cmd, capture_output=True, text=True)
# 
#         # Show console output (useful when --debug is on or errors occur)
#         with st.expander("Show console output", expanded=False):
#             st.text(result.stdout)
#             st.error(result.stderr) if result.returncode != 0 else None
# 
#         # Wait briefly to ensure file is flushed and closed
#         time.sleep(1)
# 
#         if result.returncode != 0:
#             st.error("The processing script exited with an error.")
#         elif not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
#             st.error("❌ Output video appears empty or missing.")
#         else:
#             st.success("✅ Processing complete!")
#             # Read bytes to avoid browser gray player issue
#             with open(output_path, "rb") as f:
#                 data = f.read()
#             st.video(data)
#             st.download_button(
#                 label="Download annotated video",
#                 data=data,
#                 file_name="annotated.mp4",
#                 mime="video/mp4"
#             )

# Footer help
with st.expander("What each option does"):
    st.markdown(
        """
- **Angles**: choose which on-body metrics to draw  
  - shin angle, knee flex, knee_pos (behind/over/in front of toes), elbow flex, hip flex, torso lean, head tilt, hand (wrist) flex  
- **Show overlay**: adds a summary panel with all active values  
- **Overlay position**: top/bottom & left/right corner  
- **Show skeleton**: draws MediaPipe’s white skeleton for context  
- **Font/Line/Dot sizes**: control label readability and line aesthetics  
- **Verbose logs**: print per-frame values to console  
- **Desktop-only preview**: opens OpenCV windows (not supported on Cloud)
        """
# - **Smoothing**: rolling average window (frames) for steadier numbers  
    )
