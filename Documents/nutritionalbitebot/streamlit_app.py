"""
Streamlit Frontend for Pantry-to-Plate
======================================
A minimal interactive UI that lets users:
1. Upload or select a patient profile JSON
2. Generate clinical nutrition constraints via the HierarchicalClinicalRulesEngine
3. Display a human-readable constraints summary
4. Optionally run the full Pantry-to-Plate pipeline (time-consuming)
"""

import json
from pathlib import Path
from typing import Dict
from dataclasses import asdict

import streamlit as st

# Local project imports
from clinical_rules_engine import HierarchicalClinicalRulesEngine
from main_integration import PantryToPlateSystem

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def load_sample_patient() -> Dict:
    """Load the first sample patient profile bundled with the repo."""
    sample_file = Path(__file__).with_name("sample_output.json")
    if sample_file.exists():
        with open(sample_file, "r") as f:
            data = json.load(f)
        return data["user_profiles"][0]
    st.error("sample_output.json not found – please provide a patient file.")
    st.stop()


def display_constraint_summary(engine: HierarchicalClinicalRulesEngine, constraint):
    """Render the summary text nicely in Streamlit."""
    summary_text = engine.generate_constraint_summary(constraint)
    st.download_button(
        "Download JSON",
        data=json.dumps(asdict(constraint), default=str, indent=2),
        file_name=f"clinical_constraint_{constraint.user_id}.json",
        mime="application/json",
    )
    st.code(summary_text)


def main():
    st.set_page_config(page_title="Pantry-to-Plate", layout="wide")
    st.title("🥗 Pantry-to-Plate – Clinical Meal Planning")

    # Sidebar – Patient input --------------------------------------------------
    st.sidebar.header("Patient Profile Input")
    patient_file = st.sidebar.file_uploader("Upload patient JSON", type="json")

    if patient_file is not None:
        patient_profile = json.load(patient_file)
        st.sidebar.success("Patient profile uploaded.")
    else:
        st.sidebar.info("Using bundled sample patient (click to override).")
        patient_profile = load_sample_patient()

    # Main tabs ----------------------------------------------------------------
    tabs = st.tabs(["Generate Constraints", "Run Full Pipeline"])

    # --- Tab 1: Constraints ---------------------------------------------------
    with tabs[0]:
        st.header("1️⃣ Clinical Nutrition Constraints")
        if st.button("Generate Constraints", key="btn_constraints"):
            engine = HierarchicalClinicalRulesEngine()
            constraint = engine.generate_clinical_constraints(patient_profile)

            # Display key metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Potassium daily max", f"{constraint.potassium.daily_max} {constraint.potassium.unit}")
            col2.metric("Sodium daily max", f"{constraint.sodium.daily_max} {constraint.sodium.unit}")
            col3.metric("Protein / meal", f"{constraint.protein.per_meal_protein_g} g")

            st.subheader("Prohibited Foods")
            if constraint.prohibited_foods:
                st.write("\n".join(f"• {f.food_name}" for f in constraint.prohibited_foods))
            else:
                st.write("✅ None")

            st.subheader("Constraint Summary")
            display_constraint_summary(engine, constraint)

    # --- Tab 2: Full pipeline -------------------------------------------------
    with tabs[1]:
        st.header("2️⃣ End-to-End Pipeline Execution")
        st.caption("Runs extraction, rules engine, pantry inventory, recipe generation, and report.")

        # Configuration defaults mirrored from main_integration.py
        mimic_path = st.text_input("MIMIC-IV root path", 
                                   r"C:\\Users\\pradh\\Downloads\\mimic-iv-clinical-database-demo-2.2\\mimic-iv-clinical-database-demo-2.2")
        usda_key = st.text_input("USDA API key (optional)", "")
        output_dir = st.text_input("Output directory", "outputs")

        if st.button("Run Full Pipeline", key="btn_pipeline"):
            st.info("Pipeline running… this may take a few minutes.")
            system = PantryToPlateSystem({
                "mimic_path": mimic_path,
                "usda_api_key": usda_key or None,
                "output_dir": output_dir,
            })
            results = system.run_full_pipeline(patient_id=patient_profile.get("user_id"))

            if results.get("status") == "SUCCESS":
                st.success("Pipeline completed successfully.")
                st.json(results)
            else:
                st.error(f"Pipeline failed: {results.get('error')}")


if __name__ == "__main__":
    main()
