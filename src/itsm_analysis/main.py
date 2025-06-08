# grafana_api.py
import pandas as pd
from flask import Flask, jsonify
from itsm_analysis.graphs.graph import load_features_as_dataframe, run_analysis  # Import from main_graph.py


app = Flask(__name__)


def convert_to_serializable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    elif isinstance(obj, pd.Series):
        return obj.apply(lambda x: x.isoformat() if pd.notna(x) else None).tolist()
    elif isinstance(obj, pd.DataFrame):
        result = {}
        for col in obj.columns:
            if pd.api.types.is_datetime64_any_dtype(obj[col].dtype):
                result[col] = obj[col].apply(lambda x: x.isoformat() if pd.notna(x) else None).tolist()
            else:
                result[col] = obj[col].tolist()
        return result
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    else:
        return obj


@app.route("/dashboard_data", methods=["GET"])
def get_dashboard_data():
    # Load features using the function from main_graph.py
    df = load_features_as_dataframe()


    # Run the analysis using the function from main_graph.py
    analysis_result = run_analysis(df)


    # Extract the relevant dashboard output
    dashboard_output = analysis_result.get("dashboard_output", analysis_result)


    print(f"cooomeDashboard Output: {dashboard_output}")


    #dashboard_output = analysis_result.get("categorization", analysis_result)


    #print(f"cooomecategorization: {dashboard_output["CategorizationAgent"]}")


    #dashboard_output = analysis_result.get("sla_priority_analysis", analysis_result)


    #print(f"cooomesla_priority_analysis: {dashboard_output["sla_priority_analysis"]}")


    # Convert any DataFrames within the output to a serializable format
    serializable_output = convert_to_serializable(dashboard_output)


    final_result = jsonify(serializable_output)


    return final_result


if __name__ == "__main__":
    app.run(debug=True, port=5000)