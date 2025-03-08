import streamlit as st
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class BiasMitigationTool:
    def __init__(self):
        """
        Initialize the Bias Mitigation Tool.
        """
        self.client = None

    @staticmethod
    def preprocess_target_column(target_column):
        """
        Preprocess the target column to ensure it's discrete if it contains continuous values.
        """
        if target_column.dtype in ['float64', 'int64']:
            # Bin continuous target into classes
            binned_target = pd.qcut(target_column, q=2, labels=[0, 1])  # Discretize into two classes
            return binned_target.astype(int)
        return target_column

    def analyze_bias(self, dataset, target_column, sensitive_attribute, model):
        """
        Analyze bias in terms of demographic parity difference and equalized odds.
        """
        try:
            # Preprocess target column
            dataset[target_column] = self.preprocess_target_column(dataset[target_column])

            # Prepare data
            X = dataset.drop(columns=[target_column])
            y = dataset[target_column]
            sensitive_features = dataset[sensitive_attribute]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the selected model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Store in session state
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = y_pred

            # Calculate bias metrics
            demo_parity_diff = demographic_parity_difference(
                y_test, y_pred, sensitive_features=sensitive_features.loc[y_test.index]
            )
            eq_odds_diff = equalized_odds_difference(
                y_test, y_pred, sensitive_features=sensitive_features.loc[y_test.index]
            )

            return demo_parity_diff, eq_odds_diff, y_pred, X_test, y_test
        except Exception as e:
            st.error(f"❌ Error in bias analysis: {e}")
            return None, None, None, None, None

    def mitigate_bias(self, dataset, target_column, sensitive_attribute, model):
        """
        Apply bias mitigation using Fairlearn.
        """
        try:
            # Preprocess target column
            dataset[target_column] = self.preprocess_target_column(dataset[target_column])

            # Prepare data
            X = dataset.drop(columns=[target_column])
            y = dataset[target_column]
            sensitive_features = dataset[sensitive_attribute]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Bias mitigation with ExponentiatedGradient
            mitigator = ExponentiatedGradient(estimator=model, constraints=DemographicParity())
            mitigator.fit(X_train, y_train, sensitive_features=sensitive_features.loc[X_train.index])

            mitigated_predictions = mitigator.predict(X_test)

            # Store in session state
            st.session_state["y_test"] = y_test
            st.session_state["y_pred"] = mitigated_predictions

            accuracy = accuracy_score(y_test, mitigated_predictions)

            return mitigator, mitigated_predictions, accuracy, X_test, y_test
        except Exception as e:
            st.error(f"❌ Error in bias mitigation: {e}")
            return None, None, None, None, None


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred:0", "Pred:1"], yticklabels=["True:0", "True:1"])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)


def main():
    st.markdown("<h1 class='main-title'>⚖️ Recruitment Bias Mitigation Tool</h1>", unsafe_allow_html=True)
    st.write("🔎 Upload a dataset and analyze recruitment bias for mitigation.")

    with st.sidebar:
        st.markdown("<h2 class='sidebar-title'>⚙️ Configuration</h2>", unsafe_allow_html=True)

    st.markdown("<h2 class='subtitle'>📊 Upload Your Dataset</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")

    if uploaded_file:
        dataset = pd.read_csv(uploaded_file)
        st.markdown("### 📝 Dataset Preview")
        st.dataframe(dataset)  # Improved tabular and interactive dataset preview

        st.markdown("### 📑 Dataset Summary")
        st.write(f"Number of rows: {dataset.shape[0]}")
        st.write(f"Number of columns: {dataset.shape[1]}")
        st.write(f"Column types:\n{dataset.dtypes}")

        sensitive_attribute = st.selectbox("Select Sensitive Attribute", dataset.columns)
        target_column = st.selectbox("Select Target Column", dataset.columns)

        # Model selection
        model_type = st.selectbox("Select Machine Learning Model", ["Random Forest", "Logistic Regression", "SVM"])
        if model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        elif model_type == "SVM":
            model = SVC(probability=True, random_state=42)

        if st.button("🔍 Analyze Bias"):
            with st.spinner("Analyzing bias..."):
                tool = BiasMitigationTool()
                demo_parity, eq_odds, y_pred, X_test, y_test = tool.analyze_bias(dataset, target_column, sensitive_attribute, model)
                if demo_parity is not None and eq_odds is not None:
                    st.success(f"✅ Bias Analysis Complete:")
                    st.write(f" - *Demographic Parity Difference*: {demo_parity:.4f}")
                    st.write(f" - *Equalized Odds Difference*: {eq_odds:.4f}")

                    # Plot the metrics as a bar chart
                    fig, ax = plt.subplots()
                    metrics = {'Demographic Parity Difference': demo_parity, 'Equalized Odds Difference': eq_odds}
                    ax.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange'])
                    ax.set_title("Bias Metrics")
                    ax.set_ylabel("Difference")
                    st.pyplot(fig)

                    # Additional visualizations
                    plot_confusion_matrix(y_test, y_pred)
                    plot_roc_curve(y_test, y_pred)

        if st.button("🛠️ Mitigate Bias"):
            with st.spinner("Mitigating bias..."):
                tool = BiasMitigationTool()
                mitigator, mitigated_predictions, accuracy, X_test, y_test = tool.mitigate_bias(dataset, target_column, sensitive_attribute, model)
                if mitigator:
                    st.success(f"✅ Bias Mitigation Complete. Model Accuracy: {accuracy:.4f}")

                    # Plot the mitigated confusion matrix
                    plot_confusion_matrix(y_test, mitigated_predictions)
                    plot_roc_curve(y_test, mitigated_predictions)

        if st.button("💾 Download Results"):
            if "y_test" in st.session_state and "y_pred" in st.session_state:
                result_df = pd.DataFrame({
                    "True Values": st.session_state["y_test"],
                    "Predictions": st.session_state["y_pred"]
                })
                result_csv = result_df.to_csv(index=False).encode('utf-8')  # Encode the CSV as bytes
                st.download_button("Download CSV", result_csv, "results.csv", "text/csv")
            else:
                st.warning("⚠️ You need to analyze or mitigate bias first to download the results.")


if __name__ == "__main__":
    main()