
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# EMOTION DATA
# ============================================================

EMOTION_EMOJIS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

EMOTION_COLORS = {
    "angry": "#ef4444",
    "disgust": "#22c55e",
    "fear": "#a855f7",
    "happy": "#facc15",
    "sad": "#3b82f6",
    "surprise": "#f97316",
    "neutral": "#9ca3af"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2.2rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #111827,
            #1e293b
        );
        border: 1px solid #374151;
        margin-bottom: 2rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #cbd5e1;
    }

    .result-card {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        background: #111827;
        border: 1px solid #374151;
    }

    .result-label {
        font-size: 1rem;
        color: #94a3b8;
    }

    .result-emotion {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.7rem 0;
    }

    .result-confidence {
        font-size: 1.3rem;
        color: #cbd5e1;
    }

    .info-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: #111827;
        border: 1px solid #374151;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            😊 AI Emotion Detector
        </div>

        <div class="hero-subtitle">
            Real-time facial emotion analysis using
            Computer Vision, DeepFace and Artificial Intelligence.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Detection Settings")

    input_method = st.radio(
        "Choose input method",
        [
            "📷 Camera",
            "🖼️ Upload Image"
        ]
    )

    st.divider()

    st.subheader("🧠 AI Technology")

    st.write("• DeepFace")
    st.write("• OpenCV")
    st.write("• Facial Expression Analysis")
    st.write("• Python")
    st.write("• Streamlit")

    st.divider()

    st.subheader("😊 Detectable Emotions")

    for emotion in [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral"
    ]:

        st.write(
            f"{EMOTION_EMOJIS[emotion]} "
            f"{emotion.capitalize()}"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Detection History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# INPUT
# ============================================================

image_file = None


if input_method == "📷 Camera":

    st.subheader("📷 Capture Your Expression")

    st.write(
        "Allow camera access and capture a clear image "
        "of your face."
    )

    # Removed resolution="720p" for compatibility
    image_file = st.camera_input(
        "Take a picture"
    )


else:

    st.subheader("🖼️ Upload an Image")

    image_file = st.file_uploader(
        "Choose an image containing a face",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


# ============================================================
# ANALYSIS
# ============================================================

if image_file is not None:

    st.divider()

    image_bytes = image_file.getvalue()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        st.error(
            "❌ Could not read the selected image."
        )

        st.stop()


    # ========================================================
    # IMAGE DISPLAY
    # ========================================================

    st.subheader("🖼️ Selected Image")

    col1, col2 = st.columns([2, 1])

    with col1:

        st.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">

            ### 🔍 Analysis

            The image will be analyzed using
            DeepFace.

            The AI will estimate the probability
            of seven facial-expression categories.

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # DEEPFACE
    # ========================================================

    st.subheader("🧠 AI Emotion Analysis")

    with st.spinner(
        "Analyzing facial expression..."
    ):

        try:

            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=True,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]


            # =================================================
            # RESULT
            # =================================================

            dominant_emotion = result[
                "dominant_emotion"
            ]

            emotion_scores = result[
                "emotion"
            ]

            confidence = float(
                emotion_scores[dominant_emotion]
            )

            emoji = EMOTION_EMOJIS.get(
                dominant_emotion,
                "😊"
            )


            # =================================================
            # SAVE HISTORY
            # =================================================

            st.session_state.history.append(
                {
                    "Emotion":
                        dominant_emotion.capitalize(),

                    "Confidence":
                        round(
                            confidence,
                            2
                        )
                }
            )

            # Keep only latest 10
            st.session_state.history = (
                st.session_state.history[-10:]
            )


            # =================================================
            # RESULT CARD
            # =================================================

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        DOMINANT EMOTION
                    </div>

                    <div class="result-emotion">
                        {emoji}
                        {dominant_emotion.upper()}
                    </div>

                    <div class="result-confidence">
                        Confidence:
                        <strong>
                            {confidence:.2f}%
                        </strong>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")


            # =================================================
            # METRICS
            # =================================================

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🎯 Dominant Emotion",
                    dominant_emotion.capitalize()
                )

            with col2:

                st.metric(
                    "📊 Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:

                st.metric(
                    "😊 Emotions Detected",
                    "7"
                )


            st.divider()


            # =================================================
            # CHART
            # =================================================

            st.subheader(
                "📊 Emotion Confidence Distribution"
            )

            chart_data = pd.DataFrame(
                {
                    "Emotion": [
                        emotion.capitalize()
                        for emotion in emotion_scores.keys()
                    ],

                    "Confidence": [
                        float(score)
                        for score in emotion_scores.values()
                    ]
                }
            )

            chart_data = chart_data.sort_values(
                "Confidence",
                ascending=False
            )

            st.bar_chart(
                chart_data.set_index("Emotion"),
                y="Confidence"
            )


            # =================================================
            # DETAILED SCORES
            # =================================================

            st.subheader(
                "📋 Detailed Emotion Scores"
            )

            for emotion, score in sorted(
                emotion_scores.items(),
                key=lambda x: x[1],
                reverse=True
            ):

                emotion_emoji = EMOTION_EMOJIS.get(
                    emotion,
                    "🙂"
                )

                st.write(
                    f"{emotion_emoji} "
                    f"**{emotion.capitalize()}** — "
                    f"{float(score):.2f}%"
                )

                st.progress(
                    min(
                        max(
                            int(float(score)),
                            0
                        ),
                        100
                    )
                )


            # =================================================
            # HISTORY
            # =================================================

            st.divider()

            st.subheader(
                "🕒 Detection History"
            )

            if st.session_state.history:

                history_df = pd.DataFrame(
                    st.session_state.history
                )

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as error:

            error_message = str(error)

            if (
                "Face could not be detected"
                in error_message
            ):

                st.warning(
                    "😕 No clear face was detected.\n\n"
                    "Please look directly at the camera "
                    "and try again."
                )

            else:

                st.error(
                    "❌ Emotion analysis failed."
                )

                st.code(
                    error_message
                )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()

st.subheader(
    "ℹ️ About This Project"
)

st.markdown(
    """
    ### 🧠 How it works

    **1. Image Capture**

    The application receives an image from the
    camera or from an uploaded file.

    **2. Face Detection**

    OpenCV is used to locate the face in the image.

    **3. Emotion Analysis**

    DeepFace analyzes facial-expression patterns.

    **4. Prediction**

    The system estimates seven possible emotions:

    😠 Angry · 🤢 Disgust · 😨 Fear · 😊 Happy ·
    😢 Sad · 😲 Surprise · 😐 Neutral

    **5. Visualization**

    The results are displayed using confidence
    percentages and a visual chart.

    > ⚠️ The result is an AI prediction based on
    > visible facial-expression patterns. It does
    > not determine a person's actual internal
    > emotional state.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Emotion Detector • Built with Python, "
    "OpenCV, DeepFace and Streamlit"
)

