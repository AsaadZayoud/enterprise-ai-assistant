from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str, **kwargs) -> str:
    """
    Load a prompt template from file and inject dynamic keyword arguments.
    """
    file_path = PROMPTS_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()


    return template.format(**kwargs)