"""
Check that your Gemini key works before running the flow.

    .venv\\Scripts\\python.exe scripts\\check_gemini.py                 # uses gemini-3.8-flash
    .venv\\Scripts\\python.exe scripts\\check_gemini.py gemini-3.1-pro-preview

1. Lists the Gemini models your key can call (so you can pick a valid model id).
2. Makes one tiny structured-output call through the same GeminiLLM the pipeline uses.
Reads GEMINI_API_KEY from the environment or from the project's .env file.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_env_file() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = (p.strip() for p in line.split("=", 1))
            if value and not os.getenv(key):
                os.environ[key] = value


def main() -> None:
    load_env_file()
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.8-flash"
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("No key found. Put GEMINI_API_KEY=... in the .env file (or set it in PowerShell) and retry.")

    from google import genai
    from testgen.llm import GeminiLLM
    from testgen.models import RequirementSet

    print("Gemini models available to this key (generateContent):")
    names = []
    client = genai.Client(api_key=key)   # keep a reference: a temporary client is closed mid-pagination
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        if name.startswith("gemini") and "generateContent" in (m.supported_actions or []):
            names.append(name)
    print("  " + "\n  ".join(sorted(names)) if names else "  (none returned)")
    if model not in names:
        print(f"\nWARNING: '{model}' is not in that list - pick one of the ids above in the Model dropdown.")

    print(f"\nTest call to {model} ...")
    notes = "## Payments\n### Daily limit\n- A retail customer can transfer up to 10,000 USD per day"
    result = GeminiLLM(model=model, api_key=key).generate(
        "Extract testable requirements. Use only facts in the text.",
        f"Extract the requirements from these release notes:\n{notes}", RequirementSet, {})
    r = result.requirements[0]
    print(f"OK - Gemini returned {len(result.requirements)} requirement(s): '{r.title}' "
          f"with {len(r.acceptance_criteria)} criteria, risk={r.risk}")
    print("Your key works. Select this model in the Langflow nodes and switch Offline off.")


if __name__ == "__main__":
    main()
