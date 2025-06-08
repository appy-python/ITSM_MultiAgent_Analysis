# agents/
import json
import pandas as pd
from datetime import timedelta
from langchain.agents import Tool
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import ChatPromptTemplate


from itsm_analysis.agents.base_agent import BaseAgent
from itsm_analysis.prompts.load_prompt import load_prompt_yaml
from itsm_analysis.tools.tools import (
    check_sla_breach,
    detect_priority_inconsistencies,
    tool_definitions
)
from langchain.tools.render import render_text_description




class SLAPriorityAgent(BaseAgent):
    def __init__(self, sla_definitions=None, normalize_priority_impact=True):
        super().__init__("SLAPriorityAgent")
        self.prompt_message = load_prompt_yaml("sla")["prompt"]


        self.normalize_priority_impact = normalize_priority_impact


        # Expanded default_sla to cover all (Impact_enc, Urgency_enc) from 0 to 5
        default_sla = {}
        for impact in range(0, 6):
            for urgency in range(0, 6):
                # Use your original logic for 1-3, else fallback to None (or set a high value, e.g., 999)
                if impact in [1, 2, 3] and urgency in [1, 2, 3]:
                    if (impact, urgency) == (3, 3):
                        default_sla[(3, 3)] = 4
                    elif (impact, urgency) == (3, 2):
                        default_sla[(3, 2)] = 2
                    elif (impact, urgency) == (3, 1):
                        default_sla[(3, 1)] = 1
                    elif (impact, urgency) == (2, 3):
                        default_sla[(2, 3)] = 2
                    elif (impact, urgency) == (2, 2):
                        default_sla[(2, 2)] = 1
                    elif (impact, urgency) == (2, 1):
                        default_sla[(2, 1)] = 0.5
                    elif (impact, urgency) == (1, 3):
                        default_sla[(1, 3)] = 1
                    elif (impact, urgency) == (1, 2):
                        default_sla[(1, 2)] = 0.5
                    elif (impact, urgency) == (1, 1):
                        default_sla[(1, 1)] = 0.25
                else:
                    default_sla[(impact, urgency)] = None  # or set to a high value, e.g., 999
        self.sla_definitions = {
            k: timedelta(hours=v) for k, v in (sla_definitions or default_sla).items() if v is not None
        }


        self.tools = [Tool.from_function(**tool) for tool in tool_definitions]
        self.tool_descriptions = render_text_description(self.tools)


        self.prompt_template = ChatPromptTemplate.from_template(self.prompt_message)
        self.summary_chain = self.prompt_template | self.llm


    def normalize_scales(self, df):
        for col in ['Priority_enc', 'Impact_enc']:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].max() > 3:
                df[col] = pd.qcut(df[col], 3, labels=[1, 2, 3]).astype(int)
        return df


    def run(self, state: dict) -> dict:
        df = state.features


        '''
        if self.normalize_priority_impact:
            df = self.normalize_scales(df)
        '''


        data_dict = df.to_dict(orient="list")


        # Convert tuple keys to string keys for SLA definitions
        str_sla_definitions = {}
        for k, v in self.sla_definitions.items():
            str_sla_definitions[str(k)] = v
           
        tool_chain = RunnableMap({
            "sla_breach": lambda _: check_sla_breach.invoke({
                "data": data_dict,
                "sla_definitions": str_sla_definitions
            }),
            "inconsistencies": lambda _: detect_priority_inconsistencies.invoke({
                "data": data_dict
            })
        })


        outputs = tool_chain.invoke({})


        sla_breach_data = outputs.get("sla_breach") or {}
        breached_details = sla_breach_data.get("breached_details", [])


        # Always produce breach_distribution as a list of dicts, even if empty
        breach_distribution = []
        top_breaching = []
        time_series_data = []
        severity_distribution = []
       
        if breached_details:
            breach_df = pd.DataFrame(breached_details)
            if not breach_df.empty:
                # Priority-Impact distribution
                if {"Priority_enc", "Impact_enc"}.issubset(breach_df.columns):
                    breach_by_combo = breach_df.groupby(
                        ["Priority_enc", "Impact_enc"]
                    ).size().reset_index(name="breach_count").sort_values("breach_count", ascending=False)
                    top_breaching = breach_by_combo.head(3).to_dict(orient="records")
                    breach_distribution = breach_by_combo.to_dict(orient="records")
               
                # Time series analysis
                if "Breach_Time" in breach_df.columns:
                    breach_df["breach_date"] = pd.to_datetime(breach_df["Breach_Time"])
                    breach_df["breach_week"] = breach_df["breach_date"].dt.to_period("W").dt.start_time
                    weekly_breaches = breach_df.groupby("breach_week").size().reset_index(name="breach_count")
                    time_series_data = weekly_breaches.to_dict(orient="records")
               
                # Severity distribution
                if "Severity" in breach_df.columns:
                    severity_counts = breach_df["Severity"].value_counts().reset_index()
                    severity_counts.columns = ["severity", "count"]
                    severity_distribution = severity_counts.to_dict(orient="records")


        prompt_input = {
            "tool_descriptions": self.tool_descriptions,
            "sla_breach_percentage": sla_breach_data.get("sla_breach_percentage", 0),
            "total_breaches": sla_breach_data.get("total_sla_breaches", 0),
            "total_incidents": sla_breach_data.get("total_incidents_analyzed", 0),
            "inconsistencies_found": len(outputs.get("inconsistencies") or []),
            "top_breaching_combinations": top_breaching
        }


        summary = self.summary_chain.invoke(prompt_input)
        summary_text = summary if isinstance(summary, str) else getattr(summary, 'content', str(summary))


        # Process inconsistencies for time series if available
        inconsistency_time_series = []
        inconsistency_severity = []
        inconsistencies = outputs.get("inconsistencies") or []
       
        if inconsistencies:
            inconsistency_df = pd.DataFrame(inconsistencies)
            if "Detected_At" in inconsistency_df.columns:
                inconsistency_df["detected_date"] = pd.to_datetime(inconsistency_df["Detected_At"])
                inconsistency_df["detected_week"] = inconsistency_df["detected_date"].dt.to_period("W").dt.start_time
                weekly_inconsistencies = inconsistency_df.groupby("detected_week").size().reset_index(name="inconsistency_count")
                inconsistency_time_series = weekly_inconsistencies.to_dict(orient="records")
           
            if "Severity" in inconsistency_df.columns:
                severity_counts = inconsistency_df["Severity"].value_counts().reset_index()
                severity_counts.columns = ["severity", "count"]
                inconsistency_severity = severity_counts.to_dict(orient="records")


        return {
            "SLAPriorityAgent": {
                "summary": summary_text,
                "sla_breach": sla_breach_data if sla_breach_data else {
                    "breached_details": [],
                    "breaches": [],
                    "sla_breach_percentage": 0.0,
                    "total_incidents_analyzed": 0,
                    "total_sla_breaches": 0
                },
                "inconsistencies": inconsistencies,
                "breach_distribution": breach_distribution,
                "time_series": {
                    "breaches": time_series_data,
                    "inconsistencies": inconsistency_time_series
                },
                "severity_distribution": {
                    "breaches": severity_distribution,
                    "inconsistencies": inconsistency_severity
                }
            }
        }