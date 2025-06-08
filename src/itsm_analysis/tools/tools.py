# tools/tools.py


# tools/tools.py


from pydantic import BaseModel, conint
import pandas as pd
from typing import Dict, Any, List
from langchain.tools import tool
from datetime import timedelta




@tool
def decode_categories(data: Dict[str, Any], mappings: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Decode category and subcategory encoded integers to labels using a mapping."""
    df = pd.DataFrame(data)
   
    # Safely convert keys to int and values to string in mappings
    cat_mapping = {int(k): str(v) for k, v in mappings.get("CI_Cat", {}).items()}
    subcat_mapping = {int(k): str(v) for k, v in mappings.get("CI_Subcat", {}).items()}


    # Apply mapping with safe fallback
    df["CI_Cat"] = df["CI_Cat_enc"].map(cat_mapping).fillna("Unknown")
    df["CI_Subcat"] = df["CI_Subcat_enc"].map(subcat_mapping).fillna("Unknown")


    return df.to_dict(orient="list")


@tool
def compute_weekly_trend(data: Dict[str, Any], date_column: str = "Open_Time__") -> Dict[str, Any]:
    """Compute weekly incident trends and flag spike weeks based on 1.5x rolling average."""
    df = pd.DataFrame(data)
    df["week"] = pd.to_datetime(df[date_column]).dt.to_period("W").dt.start_time
    weekly = df.groupby("week").size().reset_index(name="incident_count")
    weekly["rolling_avg"] = weekly["incident_count"].rolling(window=3, min_periods=1).mean()
    weekly["spike"] = weekly["incident_count"] > (weekly["rolling_avg"] * 1.5)
    return weekly.to_dict(orient="records")


@tool
def get_top_n(data: Dict[str, Any], column: str, n: int = 5) -> Dict[str, int]:
    """Return the top N most frequent values in the specified column."""
    df = pd.DataFrame(data)
    top_values = df[column].value_counts().head(n).to_dict()
    return top_values


@tool
def explain_spike_weeks(spike_weeks: list) -> str:
    """Provide a natural language explanation of spike weeks."""
    if not spike_weeks:
        return "No spikes were detected in incident volume."
    return f"Spikes were detected during the following weeks: {', '.join(spike_weeks)}."


class SLADefinition(BaseModel):
    sla_definitions: Dict[str, timedelta]


@tool
def check_sla_breach(data: Dict[str, Any], sla_definitions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check whether each incident in the dataset has breached its SLA based on impact, urgency, and handle time.
    """
    df = pd.DataFrame(data)
    breaches = []
    breached_details = []
    total_incidents = len(df)
    total_sla_breaches = 0
   
    # Convert tuple keys to string keys if needed
    str_sla_definitions = {}
    for k, v in sla_definitions.items():
        if isinstance(k, tuple):
            str_key = f"{k[0]}_{k[1]}"
        else:
            str_key = str(k)
        str_sla_definitions[str_key] = v
   
    for _, row in df.iterrows():
        impact = row['Impact_enc']
        urgency = row['Urgency_enc']
        handle_time = row['Handle_Time_hrs']
        open_time_raw = row['Open_Time__']
       
        try:
            open_time = pd.to_datetime(open_time_raw)
        except Exception:
            open_time = None
       
        # Use string key format to match the tuple
        key = f"{impact}_{urgency}"
        sla_duration = str_sla_definitions.get(key)
       
        if pd.notna(handle_time) and open_time and sla_duration:
            is_breach = timedelta(hours=handle_time) > sla_duration
            if is_breach:
                total_sla_breaches += 1
                # Calculate breach severity based on how much the handle time exceeds SLA
                sla_hours = sla_duration.total_seconds() / 3600
                breach_ratio = handle_time / sla_hours if sla_hours > 0 else float('inf')
               
                if breach_ratio > 2.0:
                    severity = "Critical"  # More than double the SLA
                elif breach_ratio > 1.5:
                    severity = "High"      # 50-100% over SLA
                elif breach_ratio > 1.2:
                    severity = "Medium"    # 20-50% over SLA
                else:
                    severity = "Low"       # Up to 20% over SLA
               
                breached_details.append({
                    "Incident_ID": row["Incident_ID"],
                    "Priority_enc": row.get("Priority_enc"),
                    "Impact_enc": impact,
                    "Urgency_enc": urgency,
                    "Handle_Time_hrs": handle_time,
                    "SLA_Duration_hrs": sla_duration.total_seconds() / 3600,
                    "Breach_Time": open_time.isoformat() if open_time else None,
                    "Severity": severity,
                    "Breach_Ratio": round(breach_ratio, 2)
                })
           
            breaches.append({
                "Incident_ID": row["Incident_ID"],
                "breach": is_breach,
                "Breach_Time": open_time.isoformat() if open_time else None
            })
   
    sla_breach_percentage = (total_sla_breaches / total_incidents * 100) if total_incidents > 0 else 0
   
    return {
        "breaches": breaches,
        "breached_details": breached_details,
        "total_incidents_analyzed": total_incidents,
        "total_sla_breaches": total_sla_breaches,
        "sla_breach_percentage": sla_breach_percentage
    }


@tool
def detect_priority_inconsistencies(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify incidents where the assigned priority appears inconsistent with the given impact and urgency levels.
    """
    df = pd.DataFrame(data)
    # DEBUG: Print columns and first row to diagnose data issues
    print("DEBUG: DataFrame columns:", df.columns.tolist())
    print("DEBUG: First row:", df.iloc[1].to_dict() if not df.empty else "Empty DataFrame")
    inconsistencies = []
    for _, row in df.iterrows():
        impact, urgency, priority = row['Impact_enc'], row['Urgency_enc'], row['Priority_enc']
        detected_at = None
        if "Open_Time__" in row and pd.notna(row["Open_Time__"]):
            try:
                detected_at = pd.to_datetime(row["Open_Time__"]).isoformat()
            except Exception:
                detected_at = None
        if (impact == 3 and urgency == 1 and priority != 1) or (impact == 1 and urgency == 3 and priority != 3):
            # Calculate severity based on impact and urgency
            # Higher impact and urgency = higher severity
            severity = "High" if impact == 3 and urgency == 3 else \
                      "Medium" if (impact == 3 and urgency == 2) or (impact == 2 and urgency == 3) else \
                      "Low"
           
            inconsistencies.append({
                "Incident_ID": row["Incident_ID"],
                "Impact": impact,
                "Urgency": urgency,
                "Priority": priority,
                "Description": "Potential misalignment between Impact, Urgency, and Priority.",
                "Detected_At": detected_at,
                "Severity": severity
            })
    return inconsistencies


@tool
def sla_breach_summary(breaches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize the SLA breach results by calculating the total number of incidents, number of breaches, and breach percentage.
    """
    total = len(breaches)
    breached_count = sum(1 for b in breaches if b["breach"])
    breach_pct = (breached_count / total * 100) if total > 0 else 0
    return {
        "total_incidents": total,
        "total_breaches": breached_count,
        "breach_percentage": breach_pct
    }




# Tool metadata
tool_definitions = [
    {
        "func": decode_categories,
        "name": "decode_categories",
        "description": "Maps encoded categories to their corresponding labels using predefined mappings."
    },
    {
        "func": compute_weekly_trend,
        "name": "compute_weekly_trend",
        "description": "Computes weekly incident trends and identifies spike weeks using a rolling average."
    },
    {
        "func": get_top_n,
        "name": "get_top_n",
        "description": "Returns the top N categories or subcategories based on frequency."
    },
    {
        "func": explain_spike_weeks,
        "name": "explain_spike_weeks",
        "description": "Explains spike weeks based on incident data."
    },
        {
        "func": check_sla_breach,
        "name": "check_sla_breach",
        "description": "Checks incidents for SLA breaches based on impact, urgency, and handle time."
    },
    {
        "func": detect_priority_inconsistencies,
        "name": "detect_priority_inconsistencies",
        "description": "Finds incidents with priority inconsistent with impact and urgency."
    },
    {
        "func": sla_breach_summary,
        "name": "sla_breach_summary",
        "description": "Summarizes SLA breach statistics."
    }
]