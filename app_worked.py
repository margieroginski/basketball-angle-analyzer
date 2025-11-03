import streamlit as st
import subprocess
import os
from pathlib import Path

st.set_page_config(page_title="Basketball Angle Analyzer", layout="centered")

st.title("🏀 Basketball Angle Analyzer")
st.write("Upload a side-view video and choose which body angles to annotate.")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload your .mp4 or .mov video", type=["mp4", "mov"])
input_path = "input.mp4"
if uploaded_file:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        suffix = ".mp4"
    input_path = f"input{suffix}"

    for candidate in {"input.mp4", "input.mov"} - {input_path}:
        if os.path.exists(candidate):
            os.remove(candidate)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ Uploaded: {uploaded_file.name}")

# --- Options ---
st.subheader("Options")

angles = st.multiselect(
    "Select angles to display",
    ["shin", "knee", "knee_pos", "hip", "torso", "elbow", "hand", "head"],
    default=["shin", "knee", "knee_pos", "hip", "torso", "elbow", "hand"]
)

show_overlay = st.checkbox("Show overlay panel", True)
overlay_position = st.selectbox(
    "Overlay position",
    ["top-right", "top-left", "bottom-right", "bottom-left"],
    index=0
)
show_skeleton = st.checkbox("Show skeleton (white lines)")
smoothing = st.slider("Smoothing (frames)", 1, 15, 5)

st.divider()

# --- Run processing ---
if st.button("Process Video"):
    if not uploaded_file:
        st.warning("Please upload a video first.")
    else:
        output_path = "annotated.mp4"

        # Build the command
        cmd = [
            "python", "annotate_angles.py",
            "--input", input_path,
            "--output", output_path,
            "--angles", ",".join(angles),
            "--smoothing", str(smoothing),
            "--overlay-position", overlay_position
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
#        time.sleep(1)

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


    
