# src/itsm_analysis/graphs/main_graph.py


import pandas as pd
from feast import FeatureStore
from langgraph.graph import StateGraph
from itsm_analysis.agents.sla_agent import SLAPriorityAgent
from itsm_analysis.agents.categorization_agent import CategorizationAgent
from graphs.state import AgentGraphState  # Import the state model


def load_features_as_dataframe():
    # Initialize Feast FeatureStore (assumes feature_store.yaml is in the root directory)
    store = FeatureStore(repo_path="src/itsm_analysis/feature_repo")


    # Load the entity dataframe with incident IDs and timestamps
    entity_df = pd.read_parquet(
        "src/itsm_analysis/feature_repo/data/itsm_cleaned.parquet"
    )


    print(f"Data type of Open_Time column after loading: {entity_df['Open_Time'].dtype}")
    print(f"First few values of Open_Time: {entity_df['Open_Time'].head()}")


    # Convert 'Open_Time' to datetime (UTC)
    entity_df['Open_Time'] = pd.to_datetime(entity_df['Open_Time'], utc=True, errors="coerce")


    print(f"Data type of Open_Time column before conversion: {entity_df['Open_Time'].dtype}")


    if "event_timestamp" not in entity_df.columns:
        entity_df["event_timestamp"] = entity_df["Open_Time"]
    else:
        if pd.api.types.is_datetime64_any_dtype(entity_df['event_timestamp']):
            entity_df["event_timestamp"] = entity_df["event_timestamp"].dt.tz_convert('UTC')
        else:
            entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True, errors="coerce")


    print(f"Data type of Open_Time column after loading: {entity_df['Open_Time'].dtype}")
    print(f"First few values of Open_Time: {entity_df['Open_Time'].head()}")
    print(f"Data type of event_timestamp column: {entity_df['event_timestamp'].dtype}")
    print(f"First few values of event_timestamp: {entity_df['event_timestamp'].head()}")


    # Retrieve features from Feast feature store
    feature_vector = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "itsm_features:Handle_Time_hrs",
            "itsm_features:CI_Name_enc",
            "itsm_features:CI_Cat_enc",
            "itsm_features:CI_Subcat_enc",
            "itsm_features:Closure_Code_enc",
            "itsm_features:Priority_enc",
            "itsm_features:Impact_enc",
            "itsm_features:Urgency_enc",
            "itsm_features:Resolution_SLA_Breach",
            "itsm_features:Has_KB",
        ]
    ).to_df()


    return feature_vector


def run_all_agents(state: dict) -> dict:
    sla_output = SLAPriorityAgent().run(state)
    cat_output = CategorizationAgent().run(state)


    # Merge dashboard outputs from both agents
    dashboard_output = {}
    dashboard_output.update(sla_output.get("dashboard_output", sla_output))
    dashboard_output.update(cat_output.get("dashboard_output", cat_output))


    # Convert state model to dict, then merge
    state_dict = state.model_dump()


    # Return the combined state and dashboard output for Grafana
    return {**state_dict, "dashboard_output": dashboard_output}


def build_agent_graph():
    graph = StateGraph(AgentGraphState)


    # Add a single node that runs all agents and collates output
    graph.add_node("run_agents", run_all_agents)


    # Set the entry and finish point of the graph
    graph.set_entry_point("run_agents")
    graph.set_finish_point("run_agents")


    return graph.compile()


def run_analysis(features_df: pd.DataFrame):
    # Build and compile LangGraph agent graph
    runnable = build_agent_graph()


    # Initial state to pass into the graph
    initial_state = {
        "features": features_df
    }


    # Run the graph and get output
    result = runnable.invoke(initial_state)
    print("FROM MAIN")
    print(result)
    return result


if __name__ == "__main__":
    # Load ITSM features
    df = load_features_as_dataframe()


    # Run the analysis and print the output (for direct execution)
    final_result = run_analysis(df)
    print(final_result.get("dashboard_output", final_result))  # fallback to full result