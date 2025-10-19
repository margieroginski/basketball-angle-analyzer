
#!/usr/bin/env python3

"""
annotate_angles.py

Input:  MP4 video with a single person
Output: annotated MP4 video with selectable angles and overlay

Usage:
    python annotate_angles.py --input input.mp4 --output annotated.mp4 \
        --angles shin,knee,elbow,hip,torso,head \
        --show-skeleton True --show-overlay True
"""

# get around streamlit issue with importing cv2
import os
os.system("pip install opencv-python-headless==4.10.0.84 --force-reinstall --no-cache-dir > /dev/null 2>&1")

import argparse
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


# -----------------------
# Geometry helpers
# -----------------------
def angle_between_three_points(a, b, c):
    """Return angle ABC in degrees (angle at b). a,b,c are (x,y) pixels."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6:
        return float('nan')
    cosang = np.dot(ba, bc) / (nba * nbc)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def landmark_to_px(lm, W, H):
    """mediapipe landmark -> (x,y) pixels"""
    return (int(lm.x * W), int(lm.y * H))


def nanmean(dq):
    if not dq:
        return float('nan')
    vals = [v for v in dq if v == v]
    return float(np.mean(vals)) if vals else float('nan')


# -----------------------
# Drawing helpers (dynamic sizes)
# -----------------------
def draw_segment(img, p1, p2, color, line_thickness, dot_radius):
    cv2.line(img, p1, p2, color, int(line_thickness), cv2.LINE_AA)
    cv2.circle(img, p1, int(dot_radius), color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, int(dot_radius), color, -1, cv2.LINE_AA)


def draw_text(img, text, org, color, font_scale, text_thickness):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                float(font_scale), color, int(text_thickness), cv2.LINE_AA)


# -----------------------
# Main
# -----------------------
def process_video(
    input_path,
    output_path,
    angles,
    show_skeleton=False,
    show_overlay=False,
    overlay_position = "top-right",
    smoothing=5,
    show_debug=False,
    show_debug_windows=False,
    # size controls (initial)
    font_scale=0.5,
    text_thickness=1,
    line_thickness=2,
    dot_radius=3,
):
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # old
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Use H.264 (safer for browser playback)
    # made this change when trying to get it to run with streamlit (to avoid grey video play screen)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # or 'H264' depending on your system
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    if not out.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    # Colors for each metric
    C = {
#        "shin": (255, 0, 255),      # magenta
        "shin": (255,  0, 180),     # bright pink
        "knee": (0, 255, 255),      # cyan
        "knee_pos": (0, 255, 255),  # same as knee
        "elbow": (0, 255, 0),       # green
        "hip": (0, 165, 255),       # orange-ish
        "torso": (0, 128, 255),     # teal-like
#        "head": (128, 0, 128),      # purple
        "head": (0, 215, 255),      # gold
        "hand": (255, 150, 50),     # orange
    }

    # Smoothing buffers (for numeric angles only)
    buffers = {k: deque(maxlen=max(1, smoothing)) for k in angles if k != "knee_pos"}

    # Windows
    if show_debug_windows:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        # Controls window with live sliders
        ctrl_win = "Controls"
        cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
        # cv2.createTrackbar("Font x10", ctrl_win, int(font_scale * 10), 50, lambda v: None) # ORIG
        cv2.createTrackbar("Font", ctrl_win, int(font_scale * 10), 50, lambda v: None) # NEW
        
        print(f"FONT_scale is {font_scale}")
        cv2.createTrackbar("TextThick", ctrl_win, int(text_thickness), 6, lambda v: None)
        cv2.createTrackbar("LineThick", ctrl_win, int(line_thickness), 10, lambda v: None)
        cv2.createTrackbar("DotRadius", ctrl_win, int(dot_radius), 12, lambda v: None)

    pbar = tqdm(total=nframes if nframes > 0 else None, desc="Processing frames")
    t0 = time.time()
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Read live sizes from sliders (if enabled)
            if show_debug_windows:
                fs10 = cv2.getTrackbarPos("Font", "Controls") #NEW
                # fs10 = cv2.getTrackbarPos("Font x10", "Controls") # ORIG
                tt   = cv2.getTrackbarPos("TextThick", "Controls")
                lt   = cv2.getTrackbarPos("LineThick", "Controls")
                dr   = cv2.getTrackbarPos("DotRadius", "Controls")
                # print(f"fs10  - {fs10}")
                font_scale     = max(0.1, fs10 / 10.0)

                # print(f"font_scale is {font_scale}")

                text_thickness = max(1, tt)
                line_thickness = max(1, lt)
                dot_radius     = max(1, dr)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            annotated = frame.copy()

            angle_values = {}  # for overlay (numeric angles only)
            knee_pos_label = None

            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark

                # pick side by visibility (shoulder/hip/knee/ankle)
                lv = (lms[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility +
                      lms[mp_pose.PoseLandmark.LEFT_HIP].visibility +
                      lms[mp_pose.PoseLandmark.LEFT_KNEE].visibility +
                      lms[mp_pose.PoseLandmark.LEFT_ANKLE].visibility)
                rv = (lms[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility +
                      lms[mp_pose.PoseLandmark.RIGHT_HIP].visibility +
                      lms[mp_pose.PoseLandmark.RIGHT_KNEE].visibility +
                      lms[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility)
                use_left = lv >= rv

                idx = mp_pose.PoseLandmark
                def lm_idx(L, R): return L if use_left else R

                # Points in pixels
                shoulder = landmark_to_px(lms[lm_idx(idx.LEFT_SHOULDER, idx.RIGHT_SHOULDER)], W, H)
                elbow    = landmark_to_px(lms[lm_idx(idx.LEFT_ELBOW,    idx.RIGHT_ELBOW   )], W, H)
                wrist    = landmark_to_px(lms[lm_idx(idx.LEFT_WRIST,    idx.RIGHT_WRIST   )], W, H)
                index_f  = landmark_to_px(lms[lm_idx(idx.LEFT_INDEX,    idx.RIGHT_INDEX   )], W, H)
                hip      = landmark_to_px(lms[lm_idx(idx.LEFT_HIP,      idx.RIGHT_HIP     )], W, H)
                knee     = landmark_to_px(lms[lm_idx(idx.LEFT_KNEE,     idx.RIGHT_KNEE    )], W, H)
                ankle    = landmark_to_px(lms[lm_idx(idx.LEFT_ANKLE,    idx.RIGHT_ANKLE   )], W, H)
                nose     = landmark_to_px(lms[idx.NOSE], W, H)

                # Add toe tip (FOOT_INDEX) and its visibility for a safe fallback (margie added)
                foot_idx_lm = lms[lm_idx(idx.LEFT_FOOT_INDEX, idx.RIGHT_FOOT_INDEX)]
                foot        = landmark_to_px(foot_idx_lm, W, H)
                foot_vis    = float(foot_idx_lm.visibility)
                

                # skeleton (white) if requested
                if show_skeleton:
                    mp_draw.draw_landmarks(
                        annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=int(line_thickness), circle_radius=int(max(1, dot_radius-1))),
                        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=int(line_thickness))
                    )

                # ---- SHIN (angle vs vertical) + vertical reference line ----
                if "shin" in angles:
                    # old
                    # vertical_down = (ankle[0], ankle[1] + 120)
                    # shin_angle = angle_between_three_points(knee, ankle, vertical_down)

                    vertical_up = (ankle[0], ankle[1] - 120)
                    shin_angle = angle_between_three_points(knee, ankle, vertical_up)

                    buffers["shin"].append(shin_angle)
                    shin_sm = nanmean(buffers["shin"])
                    angle_values["Shin"] = shin_sm
                    draw_segment(annotated, knee, ankle, C["shin"], line_thickness, dot_radius)
                    # vertical reference line (light gray), a bit thinner for clarity
                    # cv2.line(annotated, ankle, (ankle[0], ankle[1] - 150), (200, 200, 200),
                    #          int(max(1, line_thickness - 1)), cv2.LINE_AA)
                    draw_text(annotated, f"Shin: {shin_sm:.1f} deg", (ankle[0] + 10, ankle[1] - 10),
                              C["shin"], font_scale, text_thickness)

                # ---- KNEE angle ----
                if "knee" in angles:
                    knee_angle = angle_between_three_points(hip, knee, ankle)
                    buffers["knee"].append(knee_angle)
                    knee_sm = nanmean(buffers["knee"])
                    angle_values["Knee"] = knee_sm
                    draw_segment(annotated, hip, knee, C["knee"], line_thickness, dot_radius)
                    draw_segment(annotated, knee, ankle, C["knee"], line_thickness, dot_radius)
                    # old draw_text(annotated, f"Knee: {knee_sm:.1f} deg", (knee[0] + 10, knee[1] - 10), C["knee"], font_scale, text_thickness)
                    # new from margie
                    knee_flex = 180 - knee_sm  # Convert 180=straight → 0=straight
                    draw_text(annotated, f"Knee flex: {knee_flex:.1f} deg", (knee[0] + 10, knee[1] - 10), C["knee"], font_scale, text_thickness)
                    

                # ---- KNEE POSITION (separate option) ----
#                 if "knee_pos" in angles:
#                     tol = 5
#                     if use_left:
#                         if knee[0] < ankle[0] - tol:
#                             kp = "Behind toes"
#                         elif abs(knee[0] - ankle[0]) <= tol:
#                             kp = "Over toes"
#                         else:
#                             kp = "In front of toes"
#                     else:
#                         if knee[0] > ankle[0] + tol:
#                             kp = "Behind toes"
#                         elif abs(knee[0] - ankle[0]) <= tol:
#                             kp = "Over toes"
#                         else:
#                             kp = "In front of toes"


                # ---- KNEE POSITION (vs toe tip; fallback to ankle if toe not visible) ----

                if "knee_pos" in angles:
                    tol = 5  # pixels; tweak if needed
                    # pick reference: toe tip if visible enough, else ankle
                    ref_x = foot[0] if foot_vis >= 0.5 else ankle[0]

                    if use_left:
                        # camera left leg: smaller x is leftward on screen
                        if knee[0] < ref_x - tol:
                            kp = "Knee pos: Behind toes"
                        elif abs(knee[0] - ref_x) <= tol:
                            kp = "Knee pos: Over toes"
                        else:
                            kp = "Knee pos: In front of toes"
                    else:
                        # camera right leg: flip the comparison
                        if knee[0] > ref_x + tol:
                            kp = "Knee pos: Behind toes"
                        elif abs(knee[0] - ref_x) <= tol:
                            kp = "Knee pos: Over toes"
                        else:
                            kp = "Knee pos: In front of toes"

                    knee_pos_label = kp  # for overlay

                    if "knee" in angles:
                        # place below the knee angle label
                        text_org = (knee[0] + 10, knee[1] + 15) # was 20
                    else:
                        # place near knee even if knee angle isn't shown
                        text_org = (knee[0] + 10, knee[1] - 10)
                    draw_text(annotated, f"{kp}", text_org, C["knee_pos"], font_scale, text_thickness)

                # ---- ELBOW ----
                if "elbow" in angles:
                    # old
                    # elbow_angle = angle_between_three_points(shoulder, elbow, wrist)
                    # buffers["elbow"].append(elbow_angle)
                    # elbow_sm = nanmean(buffers["elbow"])
                    # angle_values["Elbow"] = elbow_sm
                    # draw_segment(annotated, shoulder, elbow, C["elbow"], line_thickness, dot_radius)
                    # draw_segment(annotated, elbow, wrist, C["elbow"], line_thickness, dot_radius)
                    # draw_text(annotated, f"Elbow: {elbow_sm:.1f} deg", (elbow[0] + 10, elbow[1] - 10),
                    #           C["elbow"], font_scale, text_thickness)

                    elbow_angle = angle_between_three_points(shoulder, elbow, wrist)
                    buffers["elbow"].append(elbow_angle)
                    elbow_sm = nanmean(buffers["elbow"])

                    # Convert so 0° = straight elbow, increases with flexion
                    elbow_flex = 180 - elbow_sm

                    angle_values["Elbow flex"] = elbow_flex
                    draw_segment(annotated, shoulder, elbow, C["elbow"], line_thickness, dot_radius)
                    draw_segment(annotated, elbow, wrist, C["elbow"], line_thickness, dot_radius)
                    draw_text(annotated, f"Elbow flex: {elbow_flex:.1f} deg",
#                              (elbow[0] + 10, elbow[1] - 10),
                              (elbow[0] - 40, elbow[1] - 20),
                              C["elbow"], font_scale, text_thickness)
                    
                    # ---- ELBOW HEIGHT (vertical offset vs shoulder, in pixels; + = above) ----
                    elbow_height_label = None
                    if "elbow_height" in angles:
                        # Positive if elbow is higher (smaller y) than shoulder: shoulder.y - elbow.y
                        # Note: in image coords, y grows downward; so (shoulder_y - elbow_y) > 0 means elbow above shoulder
                        dy_px = float(shoulder[1] - elbow[1])
                        buffers["elbow_height"].append(dy_px)
                        eh_sm = nanmean(buffers["elbow_height"])

                        elbow_height_label = f"Elbow height: {eh_sm:+.0f} px (vs shoulder)"

                        # Draw the text near the elbow (offset a bit above the elbow label if present)
                        draw_text(
                            annotated,
                            f"Elbow height: {eh_sm:+.0f} px",
#                            (elbow[0] + 10, elbow[1] - 50),
                            (elbow[0] -40, elbow[1] - 40),
                            C["elbow"],
                            font_scale,
                            text_thickness
                        )

                    # Optional: a light vertical guide from shoulder to elbow (comment out if you don’t want it)
                    # cv2.line(annotated, (shoulder[0], shoulder[1]), (elbow[0], elbow[1]), (180, 180, 180), max(1, line_thickness - 1), cv2.LINE_AA)
                
                    

                # ---- HIP (hip flexion) ----
                if "hip" in angles:
                    # old
                    # hip_flex = angle_between_three_points(shoulder, hip, knee)
                    # buffers["hip"].append(hip_flex)
                    # hip_sm = nanmean(buffers["hip"])
                    # angle_values["Hip flex"] = hip_sm
                    # draw_segment(annotated, shoulder, hip, C["hip"], line_thickness, dot_radius)
                    # draw_segment(annotated, hip, knee, C["hip"], line_thickness, dot_radius)
                    # draw_text(annotated, f"Hip flex: {hip_sm:.1f} deg", (hip[0] + 10, hip[1] - 10),
                    #           C["hip"], font_scale, text_thickness)

                    hip_angle = angle_between_three_points(shoulder, hip, knee)
                    buffers["hip"].append(hip_angle)
                    hip_sm = nanmean(buffers["hip"])

                    # Convert so 0 = straight torso (no flexion), larger = more flexed
                    hip_flex = 180 - hip_sm

                    angle_values["Hip flex"] = hip_flex
                    draw_segment(annotated, shoulder, hip, C["hip"], line_thickness, dot_radius)
                    draw_segment(annotated, hip, knee, C["hip"], line_thickness, dot_radius)
                    draw_text(annotated, f"Hip flex: {hip_flex:.1f} deg", (hip[0] + 10, hip[1] - 10),
                              C["hip"], font_scale, text_thickness)
                    

                # ---- TORSO (lean vs vertical) ----
                if "torso" in angles:
                    vertical_down_hip = (hip[0], hip[1] + 100)
                    # old
                    # torso_ang = angle_between_three_points(shoulder, hip, vertical_down_hip)
                    # buffers["torso"].append(torso_ang)
                    # torso_sm = nanmean(buffers["torso"])
                    # angle_values["Torso"] = torso_sm
                    # draw_segment(annotated, hip, shoulder, C["torso"], line_thickness, dot_radius)
                    # draw_text(annotated, f"Torso: {torso_sm:.1f} deg", (shoulder[0] + 10, shoulder[1] - 10),
                    #           C["torso"], font_scale, text_thickness)

                    torso_angle = angle_between_three_points(shoulder, hip, vertical_down_hip)
                    buffers["torso"].append(torso_angle)
                    torso_sm = nanmean(buffers["torso"])

                    # Convert so 0° = vertical torso, increasing with lean
                    torso_lean = 180 - torso_sm

                    angle_values["Torso"] = torso_lean
                    draw_segment(annotated, hip, shoulder, C["torso"], line_thickness, dot_radius)
                    draw_text(annotated, f"Torso lean: {torso_lean:.1f} deg",
                              (shoulder[0] + 10, shoulder[1] - 10),
                              C["torso"], font_scale, text_thickness)
                    

                # ---- HEAD (tilt) ----
                if "head" in angles:
                    vertical_up_sh = (shoulder[0], shoulder[1] - 100)
                    head_ang = angle_between_three_points(nose, shoulder, vertical_up_sh)
                    buffers["head"].append(head_ang)
                    head_sm = nanmean(buffers["head"])
                    angle_values["Head"] = head_sm
                    draw_segment(annotated, shoulder, nose, C["head"], line_thickness, dot_radius)
                    draw_text(annotated, f"Head: {head_sm:.1f} deg", (nose[0] + 10, nose[1] - 10),
                              C["head"], font_scale, text_thickness)

                # ---- HAND (wrist flexion: forearm vs hand) ----
                # old
                # if "hand" in angles:
                    # hand_ang = angle_between_three_points(elbow, wrist, index_f)
                    # buffers["hand"].append(hand_ang)
                    # hand_sm = nanmean(buffers["hand"])
                    # angle_values["Hand"] = hand_sm
                    # draw_segment(annotated, elbow, wrist, C["hand"], line_thickness, dot_radius)
                    # draw_segment(annotated, wrist, index_f, C["hand"], line_thickness, dot_radius)
                    # draw_text(annotated, f"Hand: {hand_sm:.1f} deg", (wrist[0] + 10, wrist[1] - 10),
                    #           C["hand"], font_scale, text_thickness)

                # ---- HAND (wrist flexion: forearm vs hand) ----
                if "hand" in angles:
                    hand_angle = angle_between_three_points(elbow, wrist, index_f)
                    buffers["hand"].append(hand_angle)
                    hand_sm = nanmean(buffers["hand"])

                    # Convert so 0° = straight wrist, increases with flexion
                    hand_flex = 180 - hand_sm

                    angle_values["Hand flex"] = hand_flex
                    draw_segment(annotated, elbow, wrist, C["hand"], line_thickness, dot_radius)
                    draw_segment(annotated, wrist, index_f, C["hand"], line_thickness, dot_radius)
                    draw_text(annotated, f"Hand flex: {hand_flex:.1f} deg", (wrist[0] - 50 , wrist[1] + 40),
                              C["hand"], font_scale, text_thickness,
                    )
                    

            # ---- SINGLE overlay panel (upper-right, shifted down & left) ----
            
            # ---- SINGLE overlay panel (upper-right, shifted down & left) ----
            # old
            # if show_overlay and (angle_values or knee_pos_label):
            #     lines = [f"{k}: {v:.1f} deg" for k, v in angle_values.items()]
            #     if "knee_pos" in angles and knee_pos_label:
            #         lines.append(f"{knee_pos_label}")

            #    line_h = int(20 + 8 * font_scale)
            #     shift = int(1.5 * line_h)

            #     # Measure text widths
            #     maxw = 0
            #     for s in lines:
            #         (tw, th), _ = cv2.getTextSize(
            #             s, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), int(text_thickness)
            #         )
            #         maxw = max(maxw, tw)

            #     box_w = maxw + 20
            #     top_pad = 10
            #     bottom_pad = 15
            #     box_h = top_pad + bottom_pad + line_h * len(lines)

            #     top_left = (W - box_w - 10 - shift, 10 + shift)
            #     bottom_right = (W - 10 - shift, top_left[1] + box_h)

            #     overlay = annotated.copy()
            #     cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
            #     cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

            #     # Draw lines evenly spaced inside the padded box
            #     y = top_left[1] + top_pad + line_h
            #     for s in lines:
            #         cv2.putText(
            #             annotated,
            #             s,
            #             (top_left[0] + 10, y),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             float(font_scale),
            #             (255, 255, 255),
            #             int(text_thickness),
            #             cv2.LINE_AA,
            #         )
            #         y += line_h

            # ---- SINGLE overlay panel (configurable placement) ----
            if show_overlay and (angle_values or knee_pos_label):

                lines = [f"{k}: {v:.1f} deg" for k, v in angle_values.items()]
                if "knee_pos" in angles and knee_pos_label:
                    lines.append(f"{knee_pos_label}")
                    
                # Add elbow height line if present
                if "elbow_height" in angles and elbow_height_label:
                    lines.append(elbow_height_label)
                
                line_h = int(20 + 8 * font_scale)
                shift = int(1.5 * line_h)

                # Measure text widths
                maxw = 0
                for s in lines:
                    (tw, th), _ = cv2.getTextSize(
                        s, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), int(text_thickness)
                    )
                    maxw = max(maxw, tw)

                box_w = maxw + 20
                top_pad = 10
                bottom_pad = 15
                box_h = top_pad + bottom_pad + line_h * len(lines)

                # 🔹 Overlay placement logic (based on --overlay-position)
                margin = 10
                if overlay_position == "top-right":
                    top_left = (W - box_w - margin - shift, margin + shift)
                elif overlay_position == "top-left":
                    top_left = (margin, margin + shift)
                elif overlay_position == "bottom-right":
                    top_left = (W - box_w - margin - shift, H - box_h)
                elif overlay_position == "bottom-left":
                    top_left = (margin, H - box_h)
                else:
                    top_left = (W - box_w - margin - shift, margin + shift)  # fallback
                bottom_right = (top_left[0] + box_w, top_left[1] + box_h)
                
                # margin = 10
                # if overlay_position == "top-right":
                #     top_left = (W - box_w - margin - shift, margin + shift)
                # elif overlay_position == "top-left":
                #     top_left = (margin, margin + shift)
                # elif overlay_position == "bottom-right":
                #     top_left = (W - box_w - margin - shift, H - box_h - margin - shift)
                # elif overlay_position == "bottom-left":
                #     top_left = (margin, H - box_h - margin - shift)
                # else:
                #     top_left = (W - box_w - margin - shift, margin + shift)  # fallback
                # bottom_right = (top_left[0] + box_w, top_left[1] + box_h)

                # Draw the background box
                overlay = annotated.copy()
                cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

                # Draw each text line
                y = top_left[1] + top_pad + line_h
                for s in lines:
                    cv2.putText(
                        annotated,
                        s,
                        (top_left[0] + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        float(font_scale),
                        (255, 255, 255),
                        int(text_thickness),
                        cv2.LINE_AA,
                    )
                    y += line_h


            # write / preview
            out.write(annotated)
            processed += 1
            pbar.update(1)

            if show_debug:
                if angle_values:
                    dbg = ", ".join([f"{k}={v:.1f}" for k, v in angle_values.items()])
                else:
                    dbg = "no-angles"
                if knee_pos_label:
                    dbg += f", knee_pos={knee_pos_label}"
                print(f"Frame {processed}: {dbg}")

            if show_debug_windows:
                cv2.imshow("Preview", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Early exit (q).")
                    break

    finally:
        pbar.close()
        cap.release()
        out.release()
        pose.close()
        if show_debug_windows:
            cv2.destroyAllWindows()

    dt = time.time() - t0
    fps_eff = processed / dt if dt > 0 else 0.0
    print(f"Done. {processed} frames in {dt:.2f}s ({fps_eff:.2f} fps). Saved: {output_path}")


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Annotate a side-view basketball video with body angles (colored lines + on-body labels).",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--input", required=True, help="Path to the input MP4 video.")
    parser.add_argument("--output", required=True, help="Path to save the annotated MP4 video.")

    parser.add_argument(
        "--angles",
        default="shin,knee,knee_pos,elbow,elbow_height,hip,torso,head,hand",
        help=(
            "Comma-separated list of angles to annotate.\n\n"
            "Available options:\n"
            "  shin         - Shin angle relative to vertical (draws a vertical reference line at the ankle)\n"
            "  knee         - Knee flexion angle (thigh vs shin)\n"
            "  knee_pos     - Knee position relative to toes (Behind / Over / In front). Same color as knee.\n"
            "                 If 'knee' is also selected, this shows to the RIGHT of the knee angle.\n"
            "  elbow        - Elbow flexion angle (upper vs lower arm)\n"
            "  elbow_height - Elbow vertical offset vs shoulder (px; + above / - below)\n"
            "  hip          - Hip flexion angle (torso vs thigh)\n"
            "  torso        - Torso lean angle relative to vertical (at hip)\n"
            "  head         - Head tilt relative to torso (shoulder anchor)\n"
            "  hand         - Hand flexion angle (forearm vs hand; wrist → index direction)\n\n"
            "Example:\n"
            "  --angles knee,knee_pos,hip,hand"
        )
    )

    parser.add_argument("--show-skeleton", action="store_true",
                        help="Draw the MediaPipe skeleton (white lines with dots).")
    parser.add_argument("--show-overlay", action="store_true",
                        help="Show a single semi-transparent overlay panel (upper-right, shifted down/left).")
    parser.add_argument("--debug-window", action="store_true",
                        help="Show live preview window; press 'q' to quit early.")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-frame debug values to the console.")
    parser.add_argument("--smoothing", type=int, default=5,
                        help="Temporal smoothing window size for numeric angles (frames).")

    # Size controls: small defaults restored
    parser.add_argument("--font-scale", type=float, default=0.5, help="Initial font scale for labels (default 0.5, range .1-5)")
    parser.add_argument("--text-thickness", type=int, default=1, help="Initial text thickness (default 1, range 1-6)")
    parser.add_argument("--line-thickness", type=int, default=2, help="Initial line thickness (default 2, range 1-10)")
    parser.add_argument("--dot-radius", type=int, default=3, help="Initial joint dot radius (default 3, range 1-12)")

    parser.add_argument(
        "--overlay-position",
        choices=["top-right", "top-left", "bottom-right", "bottom-left"],
        default="bottom-right",
        help="Position of the overlay panel (default: top-right)."
    )
    

    args = parser.parse_args()
    selected_angles = [a.strip().lower() for a in args.angles.split(",") if a.strip()]

    process_video(
        args.input,
        args.output,
        angles=selected_angles,
        show_skeleton=args.show_skeleton,
        show_overlay=args.show_overlay,
        overlay_position = args.overlay_position,
        smoothing=max(1, args.smoothing),
        show_debug=args.debug,
        show_debug_windows=args.debug_window,
        font_scale=args.font_scale,
        text_thickness=args.text_thickness,
        line_thickness=args.line_thickness,
        dot_radius=args.dot_radius,
    )
