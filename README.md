# LangChain + Groq Experiments
 
This repo is a set of learning notebooks that explore how to build small AI apps using **LangChain** and **Groq** (a fast LLM hosting service). Each notebook builds on a simple idea: take a prompt, send it to an LLM, and get back something useful — plain text, structured JSON, or an AI "agent" that can use tools like web search and a calculator.
 
If you're new to these terms, don't worry — this README explains everything in plain language with small code examples.
 
---
 
## What's in this repo
 
| Notebook | What it does |
|---|---|
| `feedback_intelligence_system.ipynb` | Turns messy customer feedback into clean, structured JSON (sentiment, category, issue). |
| `agent.ipynb` | Builds an AI "agent" that can search the web and do math using tools. |
| `ai_decision_analyzer.ipynb` | Compares job offers and gives a structured recommendation in JSON. |
| `langsmith_observability.ipynb` | Shows how to turn on LangSmith so you can see/debug what your LLM app is doing behind the scenes. |
 
---
 
## Key concepts (in plain language)
 
- **LLM (Large Language Model)** — the AI model itself (here, `openai/gpt-oss-120b` running on Groq's servers). It just takes text in and gives text out.
- **LangChain** — a toolkit that makes it easier to build things *around* an LLM: prompt templates, output parsers, agents, etc. Think of it as scaffolding.
- **`langchain-core`** — the basic building blocks (prompt templates, output parsers).
- **`langgraph`** — used for building more complex, multi-step AI workflows/agents.
- **LangSmith** — a dashboard that logs every LLM call your app makes, so you can debug and improve it.
- **Agent** — an LLM combined with *tools* (like a search engine or calculator) that it can decide to call on its own. Formula: **Agent = LLM + Tools**.
- **Output Parser** — takes the LLM's raw text reply and converts it into a usable format (like a JSON object or Python object).
---
 
## 1. Feedback Intelligence System
 
**Goal:** Given a piece of customer feedback like *"I love this product"*, automatically return clean JSON like:
```json
{"sentiment": "positive", "category": "product", "issue": "loves the product"}
```
 
**How it works, step by step:**
 
**Step 1 — Build a prompt template.** This is like a fill-in-the-blank message. It has a fixed "system" instruction (the rules) and a "human" part with a placeholder (`{feedback}`) that gets filled in later.
 
```python
from langchain_core.prompts import ChatPromptTemplate
 
template = ChatPromptTemplate(
    messages=[
        ("system", "You are a strict json generator. Return sentiment, category, issue as JSON only."),
        ("human", "Feedback: ```{feedback}```")
    ]
)
```
 
**Step 2 — Connect to the LLM.** `ChatGroq` is the object that actually talks to the Groq API.
 
```python
from langchain_groq import ChatGroq
 
gpt_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-120b",
    temperature=1  # 0 = very predictable, 1 = more creative/varied
)
```
 
**Step 3 — Define the exact output shape you want, using Pydantic.** Pydantic checks that the data matches the rules you set (e.g. "sentiment must be one of these three words").
 
```python
from pydantic import BaseModel
from typing import Literal
 
class feedbackoutput(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    category: Literal["product", "delivery", "support", "pricing", "other"]
    issue: str
```
 
**Step 4 — Chain everything together.** The `|` symbol pipes the output of one step into the next: template → LLM → parser.
 
```python
from langchain_core.output_parsers import PydanticOutputParser
 
json_parser = PydanticOutputParser(pydantic_object=feedbackoutput)
chain = template | gpt_llm | json_parser
 
chain.invoke({"feedback": "I just hate the delivery"})
# → feedbackoutput(sentiment='negative', category='delivery', issue='dislike of delivery')
```
 
That's it — one `chain.invoke()` call runs the whole pipeline.
 
---
 
## 2. Agent with Tools
 
**Goal:** A plain LLM can't tell you today's weather or recent news — it only knows what it learned during training. To fix that, you give it **tools** it can call when needed.
 
**Step 1 — Give the agent a search tool.**
 
```python
from langchain_community.tools import DuckDuckGoSearchRun
 
search_tool = DuckDuckGoSearchRun()
```
 
**Step 2 — Build a custom tool of your own.** Any regular Python function can become a tool by adding the `@tool` decorator (and a clear docstring — the LLM reads this to know when to use it).
 
```python
from langchain.tools import tool
 
@tool
def calculator(a: int, b: int) -> int:
    """Multiplies two numbers and returns the result."""
    return a * b
```
 
**Step 3 — Create the agent** by handing the LLM its list of tools.
 
```python
from langchain.agents import create_agent
 
agent = create_agent(model=llm, tools=[search_tool, calculator])
```
 
**Step 4 — Ask it something.** The agent decides *on its own* whether it needs a tool.
 
```python
response = agent.invoke({"messages": "what is 1234567 multiplied by 89101112?"})
# The agent calls the calculator tool automatically and returns the answer.
```
 
If you ask it about current events instead, it will call `search_tool` instead of the calculator — it picks whichever tool fits the question.
 
---
 
## 3. AI Decision Analyzer
 
**Goal:** Feed it a goal + a list of job offers in plain English, get back a structured recommendation.
 
Same pattern as notebook 1 (template → LLM → parser), but using `JsonOutputParser` (simpler, no strict schema) instead of `PydanticOutputParser`, and a more detailed system prompt that spells out the exact JSON shape it wants.
 
```python
from langchain_core.output_parsers import JsonOutputParser
 
json_parser = JsonOutputParser()
chain = template | gpt_llm | json_parser
 
chain.invoke({
    "user_input": "I got two offers: Company A ₹40k, must relocate to Pune. "
                  "Company B ₹32k, remote, better learning. My goal: work abroad."
})
```
 
This returns a full JSON object with `goal`, `options`, `decision_factors`, `recommendation`, and `reasoning` — ready to plug into a UI or dashboard.
 
---
 
## 4. LangSmith Observability
 
**Goal:** See exactly what your LangChain app is doing — every prompt sent, every response received, timing, errors — in a web dashboard.
 
**How to turn it on:** just set a few environment variables before running your chain. No code changes needed elsewhere.
 
```python
import os
 
os.environ["LANGSMITH_TRACING"] = "true"                          # turn tracing on
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = "first-langsmith-observability-project"
```
 
Once these are set, every `chain.invoke()` or `llm.invoke()` call automatically gets logged to your LangSmith project dashboard — you don't need to add any tracing code to the chain itself.
 
---
 
## Tech stack
 
- [LangChain](https://python.langchain.com/) / `langchain-core` / `langgraph` — core framework
- [Groq](https://groq.com/) via `langchain-groq` — fast LLM hosting, running `openai/gpt-oss-120b`
- [Pydantic](https://docs.pydantic.dev/) — data validation / structured output
- [LangSmith](https://smith.langchain.com/) — tracing and debugging
- `ddgs` + `langchain-community` — DuckDuckGo search tool for agents
---
 
## Setup
 
```bash
pip install -r requirements.txt
```
 
Set your API keys as environment variables — **never hardcode them in a notebook that goes on GitHub**:
 
```bash
export GROQ_API_KEY="your-groq-key"
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="your-project-name"
```
 
Or use a `.env` file (add it to `.gitignore`) with `python-dotenv`.
 
---
 
## Notes
 
These notebooks were originally exported from Google Colab and are exploratory/learning notebooks — expect some rough edges (pip install logs, iterative experimentation) rather than production-ready code.
