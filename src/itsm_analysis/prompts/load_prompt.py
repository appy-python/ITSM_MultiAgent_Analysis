import yaml


def load_prompt_yaml(prompt_name):
    with open(f'src/itsm_analysis/prompts/{prompt_name}.yaml', 'r') as f:
        return yaml.safe_load(f)