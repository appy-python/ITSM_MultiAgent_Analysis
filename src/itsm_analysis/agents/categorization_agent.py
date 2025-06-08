import json
import pandas as pd


from langchain_core.runnables import RunnableMap
from langchain.agents import Tool
from langchain_core.prompts import ChatPromptTemplate


from itsm_analysis.agents.base_agent import BaseAgent
from itsm_analysis.prompts.load_prompt import load_prompt_yaml
from itsm_analysis.tools.tools import (
    decode_categories,
    compute_weekly_trend,
    get_top_n,
    explain_spike_weeks, tool_definitions
)


from langchain.tools.render import render_text_description


class CategorizationAgent(BaseAgent):
    def __init__(self, mapping_path="data/category_mappings.json"):
        super().__init__("CategorizationAgent")
        self.mapping_path = mapping_path


        self.prompt_message = load_prompt_yaml("categorization")["prompt"]


        # Load tools from centralized tool definitions
        self.tools = [Tool.from_function(**tool) for tool in tool_definitions]
        self.tool_descriptions = render_text_description(self.tools)


        self.prompt_template = ChatPromptTemplate.from_template(self.prompt_message)
        self.summary_chain = self.prompt_template | self.llm


    def run(self, state: dict) -> dict:
        df = state.features


        print(f"[CategorizationAgent] Input DataFrame columns: {df.columns.tolist()}")


        with open(self.mapping_path) as f:
            mappings = json.load(f)


        data_dict = df.to_dict(orient="list")


        tool_chain = RunnableMap({
            "decoded": lambda _: decode_categories.invoke({"data": data_dict, "mappings": mappings}),
            "weekly": lambda _: compute_weekly_trend.invoke({"data": data_dict, "date_column": "Open_Time__"}),
            "top_categories": lambda _: get_top_n.invoke({"data": data_dict, "column": "CI_Cat_enc", "n": 5}),
            "top_subcategories": lambda _: get_top_n.invoke({"data": data_dict, "column": "CI_Subcat_enc", "n": 5})
        })


        outputs = tool_chain.invoke({})


        spike_weeks = [str(entry["week"]) for entry in outputs["weekly"] if entry.get("spike")]
        spike_explanation = explain_spike_weeks.invoke({"spike_weeks": spike_weeks})


        return {
            "CategorizationAgent": {
                "top_categories": outputs["top_categories"],
                "top_subcategories": outputs["top_subcategories"],
                "weekly_trend": outputs["weekly"],
                "spike_explanation": spike_explanation
            }
        }