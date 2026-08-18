# LangChain + Groq Experiments

A collection of Colab notebooks exploring LangChain, Groq-hosted LLMs (GPT-OSS-120B), structured output parsing, tool-calling agents, and LLM observability with LangSmith.

## Contents

| Notebook | What it covers |
|---|---|
| `feedback_intelligence_system.ipynb` | Building a strict JSON-output pipeline with `ChatPromptTemplate` → `ChatGroq` → `PydanticOutputParser` to classify customer feedback into sentiment, category, and issue summary. |
| `agent.ipynb` | Turning an LLM into an agent using `create_agent`, wiring up DuckDuckGo web search for real-time info, and defining a custom `@tool`-decorated calculator function. |
| `ai_decision_analyzer.ipynb` | A career-decision engine that takes job offers + a stated goal and returns a structured JSON recommendation via `JsonOutputParser`. |
| `langsmith_observability.ipynb` | Setting up LangSmith tracing (`LANGSMITH_TRACING`, project env vars) to observe and debug LangChain runs. |

## Stack

- [LangChain](https://python.langchain.com/) / `langchain-core` / `langgraph`
- [Groq](https://groq.com/) API (`langchain-groq`) running `openai/gpt-oss-120b`
- [Pydantic](https://docs.pydantic.dev/) for output validation
- [LangSmith](https://smith.langchain.com/) for tracing/observability
- DuckDuckGo Search (`ddgs`, `langchain-community`) for agent web search

## Setup

```bash
pip install -r requirements.txt
```

Set your API keys as environment variables (never hardcode them):

```bash
export GROQ_API_KEY="your-groq-key"
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="your-project-name"
```

Or use a `.env` file (add it to `.gitignore`) with `python-dotenv`.

## Notes

These were originally exported from Google Colab and are exploratory/learning notebooks — expect some rough edges (pip install logs, iterative experimentation) rather than production-ready code.

## License

MIT (or your preferred license — update this section).
