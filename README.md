<div align="center">

# 🤖 Multi-Agent AI Demo

### Watch AI Agents Work Together in Real-Time!

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-Framework-green?style=for-the-badge)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*A visual demonstration of RAG, Agents, and GenAI concepts*

[🚀 Quick Start](#-quick-start) •
[📺 Demo](#-see-it-in-action) •
[🧠 How It Works](#-how-it-works) •
[🎓 Learn](#-what-youll-learn)

</div>

---

## 🌟 What is This?

Ever wondered how AI agents can **work together** like a team? This project shows you exactly that!

Imagine asking a complex question like:
> *"Compare our company's products with current market trends and calculate potential revenue"*

**One AI can't do it all** — but a **team of specialized AI agents** can! 

```
You ask a question
        ↓
   🎭 SUPERVISOR
   "Hmm, this needs multiple experts..."
        ↓
   ┌────┴────┬────────┐
   ↓         ↓        ↓
  📚        🌍       🧮
  RAG       WEB     MATH
 Agent     Agent   Agent
   ↓         ↓        ↓
   └────┬────┴────────┘
        ↓
   🎯 COMBINED ANSWER
```

---

## 🎬 See It In Action

### Terminal Version (Rich Visual Output)
<img src="https://img.shields.io/badge/Terminal-Demo-black?style=flat-square" alt="Terminal">

```
╔════════════════════════════ 🎭 SUPERVISOR AGENT ════════════════════════════╗
║ 🎯 Analyzing query to determine best agent(s):                              ║
║ "What are TechCorp's core values and vacation policy?"                      ║
╚═════════════════════════════════════════════════════════════════════════════╝

  📍 ROUTING DECISION: RAG
  → Query will be handled by RAG agent

╔══════════════════════════════ 📚 RAG AGENT ═════════════════════════════════╗
║ 🔍 Searching knowledge base for:                                            ║
║ "What are TechCorp's core values and vacation policy?"                      ║
╚═════════════════════════════════════════════════════════════════════════════╝
  → Found 3 relevant chunks

════════════════════════════ ✨ FINAL RESPONSE ════════════════════════════════
```

### Jupyter Notebook Version (Interactive)
<img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter" alt="Jupyter">

Perfect for step-by-step learning and presentations!

---

## 🤖 Meet The Agents

| Agent | Role | Superpower | Tech Used |
|:-----:|:-----|:-----------|:----------|
| 🎭 | **Supervisor** | Decides which agent(s) to use | GPT-4o-mini |
| 📚 | **RAG Agent** | Searches your documents | ChromaDB + Embeddings |
| 🌍 | **Web Agent** | Searches the internet | DuckDuckGo (free!) |
| 🧮 | **Math Agent** | Crunches numbers | GPT-4o-mini |
| 🎯 | **Synthesizer** | Combines all answers | GPT-4o-mini |

---

## 🚀 Quick Start

### Prerequisites
- 🐍 Python 3.9+
- 🔑 OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/multi-agent-demo.git
cd multi-agent-demo

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
export OPENAI_API_KEY='your-key-here'  # Windows: set OPENAI_API_KEY=your-key-here
```

### Run the Demo

**Option 1: Terminal Version** (cool visual output)
```bash
python multi_agent_demo.py
```

**Option 2: Jupyter Notebook** (interactive, great for learning)
```bash
pip install jupyter
jupyter notebook multi_agent_demo_notebook.ipynb
```

---

## 📁 Project Structure

```
multi-agent-demo/
│
├── 🐍 multi_agent_demo.py          # Terminal version with Rich UI
├── 📓 multi_agent_demo_notebook.ipynb  # Jupyter notebook version
├── 📋 requirements.txt             # Python packages needed
├── 📖 README.md                    # You're reading it!
│
└── 📂 knowledge_base/              # Sample documents for RAG
    ├── 📄 company_policies.txt     # Vacation, remote work policies
    ├── 📄 products.txt             # Product catalog & pricing
    └── 📄 company_about.txt        # Company mission & values
```

---

## 🧠 How It Works

### The Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
│        "What's our vacation policy and latest AI news?"         │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    🎭 SUPERVISOR AGENT                          │
│                                                                  │
│  "This query needs both internal docs AND web search..."        │
│  Decision: Route to RAG + WEB agents                            │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        ↓                                           ↓
┌───────────────────┐                   ┌───────────────────┐
│   📚 RAG AGENT    │                   │   🌍 WEB AGENT    │
│                   │                   │                   │
│ Searches:         │                   │ Searches:         │
│ • ChromaDB        │                   │ • DuckDuckGo      │
│ • Your documents  │                   │ • Real-time web   │
│                   │                   │                   │
│ Returns: Policy   │                   │ Returns: AI news  │
│ information       │                   │ articles          │
└─────────┬─────────┘                   └─────────┬─────────┘
          └─────────────────┬─────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    🎯 SYNTHESIZER                               │
│                                                                  │
│  Combines both responses into one coherent answer               │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     ✨ FINAL RESPONSE                           │
│                                                                  │
│  "Based on our company policies, you get 20 days PTO...        │
│   Meanwhile, the latest AI news shows that..."                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **RAG** | *Retrieval Augmented Generation* - AI that can search your documents first, then answer |
| **Agent** | An AI with a specific job and tools it can use |
| **Multi-Agent** | Multiple AIs working together, each doing what they're best at |
| **LangGraph** | A framework to connect agents into a workflow |
| **Embeddings** | Converting text to numbers so AI can find similar content |
| **Vector Store** | A database optimized for finding similar text (like ChromaDB) |

---

## 🎓 What You'll Learn

By exploring this project, you'll understand:

- [x] **RAG (Retrieval Augmented Generation)**
  - How to give AI access to your own documents
  - Vector embeddings and similarity search
  - Why RAG beats fine-tuning for most use cases

- [x] **Multi-Agent Systems**
  - Supervisor pattern for routing
  - Specialized agents for different tasks
  - How agents share state/memory

- [x] **LangGraph**
  - Building agent workflows as graphs
  - Conditional routing
  - State management

- [x] **Practical Skills**
  - Working with OpenAI API
  - Using ChromaDB for vectors
  - Building interactive demos

---

## 🛠️ Customize It!

### Add Your Own Documents

Drop `.txt` files in `knowledge_base/` — the RAG agent will automatically use them!

```bash
knowledge_base/
├── your_notes.txt      # Add your own!
├── meeting_minutes.txt # Add your own!
└── ...
```

### Change the AI Model

```python
# In multi_agent_demo.py, find initialize_llm() and change:
llm = ChatOpenAI(
    model="gpt-4o",  # Upgrade to GPT-4o for better results
    # model="gpt-3.5-turbo",  # Or use cheaper model
)
```

### Add a New Agent

```python
# 1. Create your agent function
def my_cool_agent(state: AgentState) -> dict:
    query = state["query"]
    # Your logic here...
    return {"my_response": result}

# 2. Add to the graph in build_agent_graph()
workflow.add_node("my_agent", my_cool_agent)

# 3. Update supervisor's routing logic
```

---

## 🐛 Troubleshooting

<details>
<summary><b>❌ "OPENAI_API_KEY not found"</b></summary>

```bash
# Mac/Linux
export OPENAI_API_KEY='sk-your-key-here'

# Windows
set OPENAI_API_KEY=sk-your-key-here
```
</details>

<details>
<summary><b>❌ SSL Certificate errors (macOS)</b></summary>

```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```
Or the demo will auto-handle it!
</details>

<details>
<summary><b>❌ Module not found</b></summary>

```bash
# Make sure venv is activated
source venv/bin/activate  # Mac/Linux
.\venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><b>❌ DuckDuckGo rate limiting</b></summary>

The web search is free but has rate limits. Wait 1-2 minutes and try again.
</details>

---

## 📚 Resources to Learn More

| Topic | Resource |
|-------|----------|
| LangChain | [docs.langchain.com](https://docs.langchain.com) |
| LangGraph | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| RAG Concepts | [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/) |
| OpenAI API | [platform.openai.com/docs](https://platform.openai.com/docs) |
| ChromaDB | [docs.trychroma.com](https://docs.trychroma.com/) |

---

## 🤝 Contributing

Found a bug? Have an idea? PRs welcome! 

1. Fork the repo
2. Create your branch (`git checkout -b feature/cool-feature`)
3. Commit changes (`git commit -m 'Add cool feature'`)
4. Push (`git push origin feature/cool-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - Use it, modify it, share it! See [LICENSE](LICENSE) for details.

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Built with ❤️ for learning AI Agents**

[LangChain](https://langchain.com) •
[LangGraph](https://langchain-ai.github.io/langgraph/) •
[OpenAI](https://openai.com) •
[ChromaDB](https://trychroma.com)

</div>
