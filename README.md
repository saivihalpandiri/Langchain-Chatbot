# 🧠 LangChain Pro Chatbot
A full-featured, smart AI assistant built with Streamlit and LangChain, offering:

Natural chat

Advanced intent classification (Chat/Web/Document/Code)

Web search (DuckDuckGo)

PDF/Word doc Q&A + smart summarization

Code generation and explanations

Document/context-aware analytics

Secure API key handling (.env support)

Export chat as PDF/TXT

Seamless Streamlit UI

# 🚀 Features
Conversational AI: Chat naturally on any topic, with insightful, structured answers.

Ask about Docs: Upload PDF or Word files—ask anything, get focused, document-grounded answers.

Summarize and Analyze Docs: Get executive summaries, key insights, and critical evaluation of uploaded documents.

Web Search: Get up-to-date, web-sourced answers when you ask about current events or news.

Code Generation & Help: Request and receive quality, well-commented code with support for multiple languages.

Intent Detection: Automatically routes your query to the right logic (chat, web, doc, code) using an LLM-based classifier plus keyword fallback.

History & Export: Full chat history with export to PDF and TXT.

Dark/Light mode: Streamlit Theming ready.

# ⚡ Quick Start
# 1. Clone this repo and enter the directory
git clone <your-repo-url>
cd <your-repo>
# 2. Install dependencies
pip install -r requirements.txt

# Recommended packages (see lang.py):
streamlit
python-dotenv
langchain
openai
fpdf
duckduckgo-search
faiss-cpu
unstructured # For Word document parsing
# 3. Set up your OpenAI API key
Create a .env file in the project root:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxx
# 4. Run the app
streamlit run lang.py

# 📄 How It Works
Secure API Handling: The app loads your OpenAI API key from .env and never exposes it in the UI or logs.
File Upload: Upload a PDF/Word file (.pdf/.docx). The app parses, segments into pages, and creates a retriever index via OpenAI embeddings and FAISS.
Intent Classification: Each input is run through an LLM-based prompt to detect if it's:
General chat
Web search
Document Q&A / summarization
Code assistance
Smart Routing:
If web search: runs DuckDuckGo, summarizes with LLM.
If doc: queries retriever-backed LLM, does contextual answering/summarizing.
If code: produces code with explanations.
Else: general conversational AI.
Rich, Structured Responses: Multiple prompt templates ensure context-appropriate, detailed, actionable answers.
Chat History: Persistent, timestamped, exportable as PDF/TXT.
Clearing & Export: Easily reset chat or export your full conversation.

# 🔧 Advanced Features
Fallback intent logic: If LLM is uncertain (<0.6 confidence), falls back to regex/keywords.
Auto-adapts to first doc Q: If user uploads a doc, next query is routed to document answering unless clearly code/web.
Multiple code languages: Detects language, merges with user query if context is ambiguous.
Robust error handling throughout.
Customizable prompts for every major function (chat, doc search, summary, code, web).

# 🛡️ Security
OpenAI API Key: Only loaded from .env, never exposed to the user interface.

Document Handling: Uploaded files are processed and cleared; ensure server security for sensitive data.

# 📝 Credits & References
Built with Streamlit
LangChain
OpenAI GPT models
DuckDuckGo Search API
FAISS
Unstructured for docx parsing

# 📃 License
MIT License (add or modify as appropriate.)

# 🙋 FAQ

# Q: Does this app show or log my OpenAI API key?
A: No. The key is loaded internally via .env—users never see it.

# Q: What formats are supported for documents?
A: PDF and DOCX (Word) files.

# Q: Is my document uploaded to any third party?
A: No—the file is only processed locally and by the LLM via API calls.
