import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import base64


# =====================================================
# Page configuration
# =====================================================

st.set_page_config(
    page_title="CMU-iPedSep",
    page_icon="🩺",
    layout="wide"
)



# =====================================================
# Load model
# =====================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "xgboost_sepsis_model.pkl"
    )

    features = joblib.load(
        "feature_names.pkl"
    )

    return model, features



model, features = load_model()



# =====================================================
# CSS
# =====================================================

st.markdown(
"""
<style>

.section-title {

    font-family: "italic";

    font-size: 30px;

    font-weight: bold;

    color:#164A7B;

}


.result-number {

    font-family:"Times New Roman";

    font-size:50px;

    font-weight:bold;

    color:#B22222;

    text-align:center;

}


body {

    font-family:"Times New Roman";

}


</style>
""",
unsafe_allow_html=True
)



# =====================================================
# Full width title banner
# =====================================================


if os.path.exists("title.png"):


    with open(
        "title.png",
        "rb"
    ) as f:

        encoded = base64.b64encode(
            f.read()
        ).decode()



    st.markdown(

    f"""

    <style>

    .title-banner img {{

        width:100%;

        height:auto;

        display:block;

    }}

    </style>


    <div class="title-banner">

    <img src="data:image/png;base64,{encoded}">

    </div>

    """,

    unsafe_allow_html=True

    )


else:


    st.warning(
        "title.png not found"
    )



st.divider()



# =====================================================
# Main layout
# =====================================================


left, right = st.columns(
    [1,1]
)



# =====================================================
# Left panel
# =====================================================


with left:


    st.markdown(

    """

    <div class="section-title">

    Patient Information Input

    </div>

    """,

    unsafe_allow_html=True

    )


    patient = {}



    patient["Age"] = st.number_input(

        "Age (months)",

        min_value=0,

        max_value=216,

        value=None

    )



    patient["Underlying condition"] = st.selectbox(

        "Underlying condition",

        options=[None,0,1],


        format_func=lambda x:

        "Please select"

        if x is None

        else

        (

            "No underlying condition"

            if x==0

            else

            "Yes (Prematurity / Chronic diseases / Severe malnutrition)"

        )

    )



    patient["Lymphocyte"] = st.number_input(

        "Lymphocyte (×10⁹/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["Monocyte"] = st.number_input(

        "Monocyte (×10⁹/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["Hemoglobin"] = st.number_input(

        "Hemoglobin (g/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["MCHC"] = st.number_input(

        "MCHC (g/L)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["RDWCV"] = st.number_input(

        "RDW-CV (%)",

        min_value=0.0,

        value=None,

        step=0.1,

        format="%.1f"

    )



    patient["NLR"] = st.number_input(

        "Neutrophil-to-Lymphocyte Ratio (NLR)",

        min_value=0.0,

        value=None,

        step=0.01,

        format="%.2f"

    )



    predict = st.button(

        "Predict Sepsis Probability"

    )



# =====================================================
# Right panel
# =====================================================


with right:


    st.markdown(

    """
    <div class="section-title">

    Prediction Output

    </div>

    """,

    unsafe_allow_html=True

    )



    if predict:


        missing=[]


        for k,v in patient.items():

            if v is None:

                missing.append(k)



        if len(missing)>0:


            st.error(

                "Prediction failed. Missing variables: "

                +

                ", ".join(missing)

            )



        else:


            X = pd.DataFrame(
                [patient]
            )


            X = X[
                features
            ]



            probability = model.predict_proba(
                X
            )[0][1]



            st.markdown(

            f"""

            <div class="result-number">

            {probability*100:.2f}%

            </div>

            """,

            unsafe_allow_html=True

            )


            st.write(

                "Predicted probability of sepsis"

            )



            st.divider()



            st.subheader(

                "Individual Feature Contribution Analysis"

            )



            booster = model.named_steps["model"]



            X_trans = model.named_steps["imputer"].transform(
                X
            )



            explainer = shap.TreeExplainer(
                booster
            )


            shap_values = explainer.shap_values(
                X_trans
            )



            fig = plt.figure(
                figsize=(8,4)
            )


            shap.plots.waterfall(

                shap.Explanation(

                    values=shap_values[0],

                    base_values=explainer.expected_value,

                    data=X_trans[0],

                    feature_names=features

                ),

                show=False

            )


            st.pyplot(fig)


# =====================================================
# Footer
# =====================================================


st.divider()


st.caption(

"""
CMU-iPedSep is an AI-assisted clinical decision support tool.
The predicted probability should be interpreted together with clinical judgement.
"""

)