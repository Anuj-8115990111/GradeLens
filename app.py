import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os

from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GradeLens",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef2ff, #f0fdf4);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    color: #475569;
}

.section {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    margin-top: 20px;
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    background-color: #f1f5f9;
    text-align: center;
    margin-top: 20px;
}

.result-number {
    font-size: 42px;
    font-weight: bold;
}

.small-text {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL PATH
# =========================================================

# IMPORTANT:
# This is deployment-friendly.
# The model file must be in the same folder as app.py.

MODEL_PATH = "smartstudy_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    try:
        model = joblib.load(MODEL_PATH)
        return model

    except Exception as e:

        st.error(
            "❌ Model could not be loaded. "
            "Make sure smartstudy_model.pkl is in the same folder as app.py."
        )

        st.stop()


model = load_model()


# =========================================================
# SHAP EXPLAINER
# =========================================================

@st.cache_resource
def load_shap_explainer(_model):

    return shap.TreeExplainer(_model)


explainer = load_shap_explainer(model)


# =========================================================
# GEMINI CLIENT
# =========================================================

@st.cache_resource
def create_gemini_client():

    # First check environment variable
    api_key = os.getenv("GEMINI_API_KEY")

    # If not available, check Streamlit Secrets
    if not api_key:

        try:
            api_key = st.secrets["GEMINI_API_KEY"]

        except Exception:

            api_key = None

    if not api_key:
        return None

    try:

        client = genai.Client(api_key=api_key)

        return client

    except Exception:

        return None


gemini_client = create_gemini_client()

GEMINI_MODEL = "gemini-3.8-flash"


# =========================================================
# TITLE / HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

<h1>🎓 GradeLens</h1>

<p>
Student Performance & Study Pattern Intelligence System
</p>

<p class="small-text">
Machine Learning • Study Pattern Analysis • Personalized Recommendations •
What-If Simulation • AI Study Assistant
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎛️ Student Information")

st.sidebar.markdown(
    "Enter the student's information below and click **Predict Student Performance**."
)


# =========================================================
# STUDY / BEHAVIOR INPUTS
# =========================================================

st.sidebar.subheader("📚 Study Information")

studytime = st.sidebar.selectbox(
    "Study Time",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "< 2 hours/week",
        2: "2–5 hours/week",
        3: "5–10 hours/week",
        4: "> 10 hours/week"
    }[x]
)

failures = st.sidebar.number_input(
    "Previous Failures",
    min_value=0,
    max_value=4,
    value=0,
    step=1
)

absences = st.sidebar.number_input(
    "Absences",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)


# =========================================================
# LIFESTYLE INPUTS
# =========================================================

st.sidebar.subheader("🧠 Lifestyle")

freetime = st.sidebar.slider(
    "Free Time",
    min_value=1,
    max_value=5,
    value=3
)

goout = st.sidebar.slider(
    "Going Out",
    min_value=1,
    max_value=5,
    value=3
)

health = st.sidebar.slider(
    "Health",
    min_value=1,
    max_value=5,
    value=3
)

traveltime = st.sidebar.slider(
    "Travel Time",
    min_value=1,
    max_value=4,
    value=1
)


# =========================================================
# SUPPORT / ACTIVITIES
# =========================================================

st.sidebar.subheader("🏫 Support & Activities")

schoolsup = st.sidebar.selectbox(
    "Extra Educational Support",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

famsup = st.sidebar.selectbox(
    "Family Educational Support",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

paid = st.sidebar.selectbox(
    "Extra Paid Classes",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

activities = st.sidebar.selectbox(
    "Extra-Curricular Activities",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

internet = st.sidebar.selectbox(
    "Internet Access",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

higher = st.sidebar.selectbox(
    "Wants Higher Education",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# =========================================================
# PREVIOUS GRADES
# =========================================================

st.sidebar.subheader("📊 Previous Grades")

G1 = st.sidebar.number_input(
    "Previous Grade G1",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)

G2 = st.sidebar.number_input(
    "Previous Grade G2",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.5
)


# =========================================================
# INPUT DATA
# =========================================================

current_input = pd.DataFrame({

    "studytime": [studytime],
    "failures": [failures],
    "absences": [absences],
    "freetime": [freetime],
    "goout": [goout],
    "health": [health],
    "traveltime": [traveltime],
    "schoolsup": [schoolsup],
    "famsup": [famsup],
    "paid": [paid],
    "activities": [activities],
    "internet": [internet],
    "higher": [higher],
    "G1": [G1],
    "G2": [G2]

})


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("""
<div class="section">

<h2>🚀 Student Performance Prediction</h2>

<p>
Enter the student's information from the sidebar and click the button below.
</p>

</div>
""", unsafe_allow_html=True)


predict_button = st.button(
    "🚀 Predict Student Performance",
    use_container_width=True
)


# =========================================================
# INITIAL STATE
# =========================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None

if "percentage" not in st.session_state:

    st.session_state.percentage = None

if "predicted_data" not in st.session_state:

    st.session_state.predicted_data = None

if "what_if_prediction" not in st.session_state:

    st.session_state.what_if_prediction = None


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    prediction = model.predict(current_input)[0]

    prediction = float(np.clip(prediction, 0, 20))

    percentage = (prediction / 20) * 100

    st.session_state.prediction = prediction

    st.session_state.percentage = percentage

    st.session_state.predicted_data = current_input.copy()

    # Reset What-If when a new prediction is made

    st.session_state.what_if_prediction = None

    # Reset chat

    if "chat_messages" in st.session_state:

        st.session_state.chat_messages = []


# =========================================================
# STOP BEFORE PREDICTION
# =========================================================

if st.session_state.prediction is None:

    st.info(
        "👈 Enter the student information from the sidebar and click "
        "**🚀 Predict Student Performance** to generate the analysis."
    )

    st.stop()


# =========================================================
# RETRIEVE STORED DATA
# =========================================================

prediction = st.session_state.prediction

percentage = st.session_state.percentage

predicted_data = st.session_state.predicted_data


# =========================================================
# PERFORMANCE LEVEL
# =========================================================

if prediction >= 15:

    performance_level = "High Performance"

elif prediction >= 10:

    performance_level = "Moderate Performance"

else:

    performance_level = "Needs Improvement"


# =========================================================
# RESULT
# =========================================================

st.markdown("""
<div class="section">

<h2>📈 Prediction Result</h2>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Predicted Final Grade",
        f"{prediction:.2f}/20"
    )


with col2:

    st.metric(
        "Approx. Percentage",
        f"{percentage:.1f}%"
    )


with col3:

    st.metric(
        "Performance Level",
        performance_level
    )


st.markdown(f"""
<div class="result-box">

<div class="result-number">
{prediction:.2f} / 20
</div>

<p>
Predicted Final Grade
</p>

<p>
Estimated Percentage: <b>{percentage:.1f}%</b>
</p>

<p>
<b>{performance_level}</b>
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# STUDY PATTERN ANALYSIS
# =========================================================

st.markdown("""
<div class="section">

<h2>🔍 Study Pattern Analysis</h2>

</div>
""", unsafe_allow_html=True)


analysis = []


if studytime <= 1:

    analysis.append(
        "📚 Study time is relatively low."
    )

elif studytime == 2:

    analysis.append(
        "📚 Study time is in the 2–5 hour/week range."
    )

elif studytime == 3:

    analysis.append(
        "📚 Study time is in the 5–10 hour/week range."
    )

else:

    analysis.append(
        "📚 Study time is above 10 hours/week."
    )


if absences > 10:

    analysis.append(
        "⚠️ Absence level is relatively high."
    )

elif absences <= 5:

    analysis.append(
        "✅ Absence level is relatively low."
    )


if failures > 0:

    analysis.append(
        f"⚠️ Previous failures recorded: {failures}."
    )

else:

    analysis.append(
        "✅ No previous failures recorded."
    )


if goout >= 4:

    analysis.append(
        "🌐 Going-out frequency is relatively high."
    )


if health <= 2:

    analysis.append(
        "❤️ Health rating is relatively low."
    )


for item in analysis:

    st.write(item)


# =========================================================
# SHAP EXPLANATION
# =========================================================

st.markdown("""
<div class="section">

<h2>🧠 Student-Specific AI Explanation</h2>

<p>
SHAP helps explain how the model's features contributed to this prediction.
These are model explanations, not causal conclusions.
</p>

</div>
""", unsafe_allow_html=True)


try:

    shap_values = explainer.shap_values(predicted_data)

    shap_values = np.array(shap_values)

    if shap_values.ndim == 2:

        shap_values = shap_values[0]

    feature_names = predicted_data.columns

    shap_df = pd.DataFrame({

        "Feature": feature_names,

        "Impact": shap_values

    })

    shap_df["Absolute Impact"] = np.abs(shap_df["Impact"])

    shap_df = shap_df.sort_values(
        "Absolute Impact",
        ascending=False
    )

    st.dataframe(
        shap_df[
            ["Feature", "Impact"]
        ].head(10),
        use_container_width=True
    )

except Exception as e:

    st.warning(
        "SHAP explanation could not be generated for this prediction."
    )


# =========================================================
# GLOBAL FEATURE IMPORTANCE
# =========================================================

st.markdown("""
<div class="section">

<h2>📊 Global Feature Importance</h2>

<p>
This shows which features were generally useful to the trained Random Forest
model across the dataset. It is not a student-specific explanation.
</p>

</div>
""", unsafe_allow_html=True)


feature_names = current_input.columns

feature_importance = model.feature_importances_


importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": feature_importance

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


st.bar_chart(
    importance_df.set_index("Feature")["Importance"]
)


# =========================================================
# PERSONALIZED RECOMMENDATIONS
# =========================================================

st.markdown("""
<div class="section">

<h2>💡 Personalized Recommendations</h2>

</div>
""", unsafe_allow_html=True)


recommendations = []


if studytime <= 2:

    recommendations.append(
        "📚 Consider increasing structured study time gradually."
    )


if absences > 10:

    recommendations.append(
        "🏫 Try to reduce unnecessary absences and maintain regular attendance."
    )


if failures > 0:

    recommendations.append(
        "📖 Focus on strengthening subjects where previous difficulties occurred."
    )


if goout >= 4:

    recommendations.append(
        "⏰ Maintain a better balance between social activities and study time."
    )


if health <= 2:

    recommendations.append(
        "❤️ Maintain healthy sleep, food and physical activity habits."
    )


if G2 < 10:

    recommendations.append(
        "🎯 Focus on improving recent academic performance and weak topics."
    )


if len(recommendations) == 0:

    recommendations.append(
        "✅ Current study pattern does not trigger any major rule-based recommendation."
    )


for recommendation in recommendations:

    st.write(recommendation)


# =========================================================
# WHAT-IF SIMULATOR
# =========================================================

st.markdown("""
<div class="section">

<h2>🔮 What-If Simulator</h2>

<p>
Change study-related variables and see how the trained model's prediction
changes. This is a model-based scenario, not a causal guarantee.
</p>

</div>
""", unsafe_allow_html=True)


what_if_studytime = st.slider(
    "What-If Study Time",
    min_value=1,
    max_value=4,
    value=studytime
)


what_if_absences = st.number_input(
    "What-If Absences",
    min_value=0,
    max_value=100,
    value=int(absences)
)


if st.button(
    "🔮 Run What-If Simulation",
    use_container_width=True
):

    what_if_input = current_input.copy()

    what_if_input["studytime"] = what_if_studytime

    what_if_input["absences"] = what_if_absences

    what_if_prediction = model.predict(
        what_if_input
    )[0]

    what_if_prediction = float(
        np.clip(
            what_if_prediction,
            0,
            20
        )
    )

    st.session_state.what_if_prediction = what_if_prediction


if st.session_state.what_if_prediction is not None:

    what_if_prediction = st.session_state.what_if_prediction

    difference = what_if_prediction - prediction


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Prediction",
            f"{prediction:.2f}"
        )


    with col2:

        st.metric(
            "What-If Prediction",
            f"{what_if_prediction:.2f}"
        )


    with col3:

        st.metric(
            "Model Difference",
            f"{difference:+.2f}"
        )


# =========================================================
# AI STUDY ASSISTANT
# =========================================================

st.markdown("""
<div class="section">

<h2>🤖 AI Study Assistant</h2>

<p>
Ask questions about the student's prediction, study pattern, recommendations,
or What-If scenario.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# FALLBACK ASSISTANT
# =========================================================

def fallback_response(question):

    question_lower = question.lower()


    if "prediction" in question_lower or "grade" in question_lower:

        return (
            f"The model predicts a final grade of "
            f"{prediction:.2f}/20 "
            f"({percentage:.1f}%)."
        )


    if "study" in question_lower:

        return (
            f"The current study-time category is {studytime}. "
            "You can use the What-If simulator to explore how changing "
            "study-related inputs affects the model prediction."
        )


    if "absence" in question_lower:

        return (
            f"The entered absence value is {absences}. "
            "The model uses absences as one of its input features."
        )


    if "recommend" in question_lower:

        return (
            "Based on the current inputs, the main recommendations are: "
            + " ".join(recommendations)
        )


    if "what if" in question_lower:

        if st.session_state.what_if_prediction is not None:

            return (
                f"The What-If scenario gives a model prediction of "
                f"{st.session_state.what_if_prediction:.2f}/20."
            )

        return (
            "Run a What-If simulation first and then ask me about the result."
        )


    return (
        "I can explain the prediction, study pattern, recommendations, "
        "feature importance, or What-If simulation."
    )


# =========================================================
# CHAT HISTORY
# =========================================================

if "chat_messages" not in st.session_state:

    st.session_state.chat_messages = []


# Display previous messages

for message in st.session_state.chat_messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask GradeLens about this student's performance..."
)


if question:

    # User message

    st.session_state.chat_messages.append({

        "role": "user",

        "content": question

    })


    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        if gemini_client is not None:

            try:

                what_if_text = "No What-If simulation has been run."

                if st.session_state.what_if_prediction is not None:

                    what_if_text = (
                        f"What-If prediction: "
                        f"{st.session_state.what_if_prediction:.2f}/20"
                    )


                prompt = f"""
You are the AI Study Assistant inside an ML project called GradeLens.

Your job is to explain the student's prediction and study pattern
in simple language.

IMPORTANT RULES:

1. Do not make causal claims.
2. Do not guarantee that changing a variable will produce a specific grade.
3. Explain that What-If results are model-based scenarios.
4. Do not invent information.
5. Distinguish global feature importance from student-specific SHAP explanations.
6. Give practical but reasonable study suggestions.
7. Keep answers understandable for a college student.

CURRENT STUDENT INFORMATION:

Predicted final grade: {prediction:.2f}/20
Estimated percentage: {percentage:.1f}%
Performance level: {performance_level}

Study time category: {studytime}
Previous failures: {failures}
Absences: {absences}
Free time: {freetime}
Going out: {goout}
Health: {health}
Travel time: {traveltime}

Extra educational support: {schoolsup}
Family support: {famsup}
Paid classes: {paid}
Activities: {activities}
Internet: {internet}
Higher education goal: {higher}

Previous grade G1: {G1}
Previous grade G2: {G2}

{what_if_text}

USER QUESTION:

{question}
"""


                response = gemini_client.models.generate_content(

                    model=GEMINI_MODEL,

                    contents=prompt

                )


                answer = response.text


            except Exception:

                answer = (
                    "Gemini was temporarily unavailable, so I will "
                    "give you a basic explanation instead.\n\n"
                    + fallback_response(question)
                )


        else:

            answer = (
                "Gemini AI is currently unavailable.\n\n"
                + fallback_response(question)
            )


        st.markdown(answer)


    st.session_state.chat_messages.append({

        "role": "assistant",

        "content": answer

    })


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "GradeLens — Student Performance & Study Pattern Intelligence System"
)

st.caption(
    "Built using Machine Learning, Random Forest, SHAP and Gemini AI."
)