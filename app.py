import streamlit as st

from chatbot import (
    get_transcript,
    create_vector_store,
    ask_question
)


st.set_page_config(
    page_title="TubeBot",
    page_icon="🎥"
)

st.title("🎥 TubeBot")

st.write(
    "Ask questions about any YouTube video."
)


# ---------------------------------------------
# SESSION STATE
# ---------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ---------------------------------------------
# YOUTUBE URL
# ---------------------------------------------

youtube_url = st.text_input(
    "Enter YouTube URL"
)


# ---------------------------------------------
# PROCESS VIDEO
# ---------------------------------------------

if st.button("Process Video"):

    # Extract ID
    video_id = youtube_url.split("v=")[-1].split("&")[0]

    with st.spinner(
        "Reading YouTube transcript..."
    ):

        transcript = get_transcript(
            video_id
        )

    with st.spinner(
        "Building knowledge base..."
    ):

        vector_store = create_vector_store(
            transcript
        )

        st.session_state.vector_store = (
            vector_store
        )

    st.success(
        "Video processed successfully! 🎉"
    )


# ---------------------------------------------
# CHAT
# ---------------------------------------------

if st.session_state.vector_store:

    question = st.chat_input(
        "Ask something about the video..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                answer = ask_question(
                    st.session_state.vector_store,
                    question
                )

                st.write(answer)