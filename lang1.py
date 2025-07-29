# ✅ Full LangChain Smart Chatbot with Pro Features
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from io import BytesIO
from fpdf import FPDF
import base64
from langchain.memory import ConversationBufferMemory
from typing import Tuple
import re
import nest_asyncio

# Apply the patch for asyncio
nest_asyncio.apply()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

if not api_key:
    st.error("❌ GOOGLE_API_KEY not found in .env file!")
    st.stop()
# Page settings
st.set_page_config(page_title="LangChain Pro Chatbot", page_icon="😶‍🌫️", layout="wide")
# Memory & LLM - Optimized for faster responses
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, max_output_tokens=4096, timeout=30)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Sidebar UI
st.sidebar.image("https://em-content.zobj.net/source/twitter/376/brain_1f9e0.png", width=40)
st.sidebar.title("LangChain Smart Chatbot")
st.sidebar.info("To change between Light and Dark mode, use the Streamlit menu (top-right) → Settings → Theme.")
# Removed mode and override_mode selection from sidebar
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.history = []
    memory.clear()
    st.rerun()

# Export chat as txt/pdf
if st.sidebar.button("📤 Export Chat") and "history" in st.session_state:
    chat_txt = "\n\n".join([f"User: {u}\nBot: {b}" for u, b, _ in st.session_state.history])
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in chat_txt.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    st.download_button("Download Chat (PDF)", pdf_bytes, file_name="chat_history.pdf")
    st.download_button("Download Chat (TXT)", chat_txt, file_name="chat_history.txt")

# Title
st.title("🧠 LangChain Smart Chatbot")
st.markdown("""
Talk, Ask about a PDF or Word file, Summarize Documents, or Search the Web — like a true pro assistant.
""")
# Improved, compact file uploader at the top with a paperclip icon
uploaded_file = st.file_uploader("📎 Upload a PDF or Word Document (optional):", type=["pdf", "docx"])
docs = []
rag_chain = None

# Track document upload in session state
if 'doc_uploaded' not in st.session_state:
    st.session_state['doc_uploaded'] = False
if 'uploaded_doc_name' not in st.session_state:
    st.session_state['uploaded_doc_name'] = None

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    with open("temp." + ext, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf") if ext == "pdf" else UnstructuredWordDocumentLoader("temp.docx")
    docs = loader.load()

    if docs:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert document analyst and AI assistant with comprehensive knowledge across all domains. The user has uploaded a document (which could be any type: academic papers, reports, articles, books, manuals, research papers, business documents, technical documentation, etc.) and is asking questions about it.

Your task is to provide COMPREHENSIVE, DETAILED, and ACCURATE answers based ONLY on the document content. Your response should:

1. **Be thorough and complete** - Cover all aspects of the question thoroughly with in-depth analysis
2. **Include relevant examples** - Provide concrete examples from the document when helpful
3. **Explain complex concepts clearly** - Break down technical terms and concepts with detailed explanations
4. **Be well-structured** - Use bullet points, paragraphs, or sections as appropriate for clarity
5. **Address the user's intent** - Understand what they're really asking for and provide targeted answers
6. **Provide context** - Give comprehensive background information and context from the document
7. **Be engaging** - Write in a helpful, conversational tone that maintains interest
8. **Be comprehensive** - Answer the question completely, not just partially
9. **Adapt to document type** - Whether it's academic, technical, business, or any other format
10. **Provide deep insights** - Offer analytical perspectives and critical thinking
11. **Include practical applications** - Show how concepts apply in real-world scenarios
12. **Address nuances** - Cover subtle differences and important distinctions

If the question is about:
- Technical concepts: Explain with detailed examples from the document, including use cases and applications
- Academic content: Provide scholarly analysis with references and theoretical frameworks
- Business information: Offer strategic insights with practical applications and market implications
- Research findings: Explain methodology, results, implications, and future research directions
- Procedures/Instructions: Give comprehensive step-by-step guidance with troubleshooting tips
- Data/Analysis: Provide detailed interpretation of findings with statistical significance
- Comparisons: Provide detailed analysis with pros/cons, similarities/differences, and use cases
- Definitions: Give comprehensive explanations with examples, etymology, and related concepts
- How-to questions: Provide detailed step-by-step instructions with best practices
- Analysis: Offer deep insights, critical thinking, and multiple perspectives
- Summaries: Provide detailed, structured summaries with key takeaways
- Specific details: Give thorough explanations with context, background, and implications

Document Context:
{context}

User Question: {question}

Provide a comprehensive, detailed answer that fully addresses the user's question based on the document content. Include specific examples, detailed explanations, and practical insights:
"""
        )

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": custom_prompt}
        )
        st.success(f"✅ {ext.upper()} loaded successfully! Found {len(docs)} pages.")
        # Add system message to chat history
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        st.session_state.history.append((f"[System] Document uploaded: {uploaded_file.name}", f"You can now ask questions about {uploaded_file.name}.", timestamp))
        st.session_state['doc_uploaded'] = True
        st.session_state['uploaded_doc_name'] = uploaded_file.name
    else:
        st.error("❌ Failed to load document content.")

# LLM-based Intent Classifier (Improved)
intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intent classifier for a smart assistant. Classify the user query into one of these intents:
- "chat": General conversation, questions, or requests for explanations.
- "web": Needs up-to-date information or web search.
- "doc": Asks about the uploaded document.
- "code": Requests code, programming help, or code explanations.

Examples:
- "Tell me a joke" → chat
- "What is the weather in Paris today?" → web
- "Summarize this document" → doc
- "Python code for bubble sort" → code
- "Show me the latest news about AI" → web
- "Explain this PDF" → doc
- "How do I reverse a string in JavaScript?" → code
- "Who won the World Cup in 2022?" → web
- "What is RAG in AI?" → web
- "Explain machine learning concepts" → web
- "How does neural networks work?" → web

Query: {query}
Intent (chat / web / doc / code):
Also provide a confidence score from 0 (very unsure) to 1 (very confident) based on how clear the intent is. Format: intent|confidence
"""
)
intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

def classify_intent(user_input: str) -> Tuple[str, float]:
    """Classify intent and return (intent, confidence)."""
    result = intent_chain.run(query=user_input).strip().lower()
    if '|' in result:
        intent, conf = result.split('|', 1)
        try:
            confidence = float(conf)
        except Exception:
            confidence = 0.5
        intent = intent.strip()
    else:
        intent = result.strip()
        confidence = 0.5
    return intent, confidence

def fallback_intent(user_input: str) -> str:
    """Fallback intent detection using keywords/regex."""
    code_keywords = [r'\bcode\b', r'\bpython\b', r'\bjava\b', r'\bc\+\+\b', r'\bfunction\b', r'\bscript\b']
    web_keywords = [r'\bnews\b', r'\bweather\b', r'\bsearch\b', r'\bcurrent\b', r'\blatest\b', r'\brag\b', r'\bmachine learning\b', r'\bai\b', r'\bartificial intelligence\b', r'\bneural network\b', r'\bdeep learning\b']
    doc_keywords = [r'\bpdf\b', r'\bdocument\b', r'\bfile\b', r'\bsummarize\b', r'\bsummary\b']
    for pat in code_keywords:
        if re.search(pat, user_input, re.IGNORECASE):
            return 'code'
    for pat in web_keywords:
        if re.search(pat, user_input, re.IGNORECASE):
            return 'web'
    for pat in doc_keywords:
        if re.search(pat, user_input, re.IGNORECASE):
            return 'doc'
    return 'chat'

# Add code generation chain
code_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a professional coding assistant. Write clear, well-commented code for the following request. If the user specifies a language, use it. If the request is ambiguous, ask for clarification.

Request: {query}

Code:
"""
)
code_chain = LLMChain(llm=llm, prompt=code_prompt)

# Chat History
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history with timestamps
for user_msg, bot_msg, timestamp in st.session_state.history:
    st.markdown(f"**[{timestamp}]**")
    with st.chat_message("user", avatar="✔️"):
        st.markdown(user_msg)
    with st.chat_message("assistant", avatar="https://i.pinimg.com/originals/de/3f/74/de3f74579c316731a89691851b4a7e08.png"):
        if "code" in user_msg.lower() or "python" in user_msg.lower() or "java" in user_msg.lower():
            st.code(bot_msg, language="python")
        else:
            st.markdown(bot_msg)

def preprocess_user_input(user_input, history):
    # If the input is short and matches a language, combine with previous
    code_languages = ['python', 'java', 'c++', 'javascript', 'c#', 'go', 'rust', 'typescript', 'php', 'ruby', 'swift']
    normalized = user_input.strip().lower()
    if len(normalized.split()) <= 3 and any(lang in normalized for lang in code_languages):
        # Find last user message that is not a language
        for prev_user_msg, _, _ in reversed(history):
            prev_norm = prev_user_msg.strip().lower()
            if not any(lang in prev_norm for lang in code_languages):
                # Merge previous question with current language
                return f"{prev_user_msg} in {user_input}"
    return user_input

# Restore is_junk_response and code detection logic

def is_junk_response(text):
    return len(text) < 30 or any(c in text for c in ['', '█', '♦', '◊'])

def is_code_query(user_input):
    code_keywords = ['code', 'python', 'java', 'c++', 'function', 'script']
    return any(kw in user_input.lower() for kw in code_keywords)

# Always show chat input at the bottom
user_input = st.chat_input("Ask about your doc, search the web, summarize, or just chat...")

# Web Search Tool (moved here to fix the error)
search_tool = DuckDuckGoSearchRun()

if user_input:
    # Preprocess for context-aware follow-up
    processed_input = preprocess_user_input(user_input, st.session_state.history)
    with st.chat_message("user", avatar="🦁"):
        st.markdown(user_input)
    with st.spinner("Thinking..."):
        # Initialize intent to None to guarantee it's always defined
        intent = None
        try:
            # Use intent classifier to route, with fallback
            intent, confidence = classify_intent(processed_input)
            # Debug print for intent/confidence
            print(f"[DEBUG] intent: {intent}, confidence: {confidence}, doc_uploaded: {st.session_state.get('doc_uploaded', False)}")
            # If a document was just uploaded, force next query to doc unless clearly web/code
            if st.session_state.get('doc_uploaded', False):
                if intent not in ["web", "code"]:
                    intent = "doc"
                    st.session_state['doc_uploaded'] = False  # Reset after first doc-related query
            if confidence < 0.6:  # Lowered threshold for better classification
                fallback = fallback_intent(processed_input)
                if fallback != intent:
                    intent = fallback
            if confidence < 0.3:  # Lowered threshold
                response = "I'm not sure what you want. Do you want to chat, search the web, ask about a document, or get code?"
                intent = None
            else:
                if intent == "web":
                    web_result = search_tool.run(processed_input)
                    summary_prompt = PromptTemplate(
                        input_variables=["web_result", "query"],
                        template="""
You are an expert AI assistant with deep knowledge across all domains. The user has asked a question that requires up-to-date information.

Your task is to provide a COMPREHENSIVE, DETAILED, and ACCURATE answer based on the web search results. Your response should:

1. **Be thorough and complete** - Cover all aspects of the question with in-depth analysis
2. **Include relevant examples** - Provide concrete examples when helpful for clarity
3. **Explain complex concepts clearly** - Break down technical terms with detailed explanations
4. **Be well-structured** - Use bullet points, paragraphs, or sections for better organization
5. **Address the user's intent** - Understand what they're really asking for and provide targeted answers
6. **Provide context** - Give comprehensive background information when relevant
7. **Be engaging** - Write in a helpful, conversational tone that maintains interest
8. **Include practical applications** - Show how concepts apply in real-world scenarios
9. **Address nuances** - Cover subtle differences and important distinctions
10. **Provide multiple perspectives** - Offer different viewpoints when applicable
11. **Include current trends** - Mention recent developments and future directions
12. **Offer actionable insights** - Provide practical takeaways and recommendations

For technical comparisons (like RAG vs DL, AI vs ML, etc.):
- Explain each concept in detail with definitions and characteristics
- Provide comprehensive comparisons with specific differences
- Include use cases and applications for each
- Discuss advantages and disadvantages
- Mention current trends and developments
- Provide practical examples and scenarios

User Query: {query}

Web Search Result:
{web_result}

Provide a comprehensive, detailed answer that fully addresses the user's question. Include specific examples, detailed explanations, practical insights, and thorough analysis:
"""
                    )
                    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                    response = summary_chain.run(web_result=web_result, query=processed_input)
                    if "No good DuckDuckGo Search Result" in web_result and is_code_query(processed_input):
                        response = code_chain.run(query=processed_input)
                elif intent == "doc" and rag_chain:
                    # If user asks for summary or insights, always use summarizer
                    if ("summarize" in processed_input.lower() or "summary" in processed_input.lower() or "insight" in processed_input.lower()):
                        summary_prompt = PromptTemplate(
                            input_variables=["text"],
                            template="""You are an expert document analyst with comprehensive knowledge across all domains. The user has uploaded a document (which could be any type: academic papers, reports, articles, books, manuals, research papers, business documents, technical documentation, etc.) and wants a detailed summary.

Your task is to provide a COMPREHENSIVE, DETAILED, and STRUCTURED summary of the document. Your summary should:

1. **Be thorough and complete** - Cover all major aspects of the document with in-depth analysis
2. **Include key insights** - Highlight important findings and takeaways with detailed explanations
3. **Provide context** - Give comprehensive background information and context
4. **Be well-structured** - Organize information clearly with logical sections and flow
5. **Include examples** - Reference specific examples from the document with detailed explanations
6. **Be actionable** - Provide actionable insights and recommendations with implementation guidance
7. **Be comprehensive** - Address all major themes and topics with thorough coverage
8. **Adapt to document type** - Whether it's academic, technical, business, or any other format
9. **Include critical analysis** - Offer analytical perspectives and critical thinking
10. **Provide practical applications** - Show how concepts apply in real-world scenarios
11. **Address implications** - Discuss broader implications and future considerations
12. **Include best practices** - Highlight recommended approaches and methodologies

Document Text:
{text}

Provide a comprehensive, detailed summary with:
1. **Executive Summary** - High-level overview with key highlights and main themes
2. **Key Points and Main Ideas** - Detailed breakdown of main concepts with explanations
3. **Important Insights and Findings** - Critical discoveries and conclusions with analysis
4. **Relevant Details and Context** - Supporting information and background with depth
5. **Actionable Takeaways** - Practical recommendations and next steps with implementation guidance
6. **Technical Analysis** - If applicable, technical details and explanations with examples
7. **Document Type Analysis** - Understanding of the document's purpose and format
8. **Critical Evaluation** - Strengths, limitations, and areas for improvement
9. **Practical Applications** - Real-world applications and use cases
10. **Future Implications** - Potential impact and future considerations

Comprehensive Summary and Insights:"""
                        )
                        summarizer = LLMChain(llm=llm, prompt=summary_prompt)
                        text = "\n".join([d.page_content for d in docs])
                        response = summarizer.run(text=text)
                    else:
                        output = rag_chain(processed_input)
                        response = output['result']
                        if is_junk_response(response):
                            # Enhanced fallback for document questions
                            doc_prompt = PromptTemplate(
                                input_variables=["question", "text"],
                                template="""You are an expert AI assistant analyzing a document. The user has uploaded a document (which could be any type: academic papers, reports, articles, books, manuals, research papers, business documents, technical documentation, etc.) and asked a question about it, but the specific context wasn't found.

Provide a comprehensive answer based on the general document content:

Document Content:
{text}

User Question: {question}

Provide a detailed, helpful response that:
1. Addresses the user's question as best as possible with comprehensive analysis
2. Explains what information is available in the document with detailed coverage
3. Suggests what specific parts might be most relevant with explanations
4. Offers to help with other document-related questions
5. Adapts to the document type (academic, technical, business, etc.)
6. Provides practical insights and applications
7. Includes critical analysis and multiple perspectives

Comprehensive Answer:"""
                            )
                            doc_chain = LLMChain(llm=llm, prompt=doc_prompt)
                            text = "\n".join([d.page_content for d in docs])
                            response = doc_chain.run(question=processed_input, text=text)
                        else:
                            response += f"\n\n📄 *Source: {st.session_state.get('uploaded_doc_name', 'Uploaded Document')}*"
                elif intent == "code":
                    response = code_chain.run(query=processed_input)
                else:
                    # Enhanced chat prompt for comprehensive answers
                    chat_prompt = PromptTemplate(
                        input_variables=["input"],
                        template="""
You are an expert AI assistant with comprehensive knowledge across all domains. The user has asked a question, and you need to provide a THOROUGH, DETAILED, and HELPFUL response.

Your response should be:

1. **Comprehensive** - Cover all aspects of the question thoroughly with in-depth analysis
2. **Accurate** - Provide correct and up-to-date information with verification
3. **Detailed** - Include relevant details, examples, and explanations with depth
4. **Well-structured** - Organize your response clearly with paragraphs, bullet points, or sections
5. **Educational** - Explain concepts clearly, especially technical ones with examples
6. **Engaging** - Write in a helpful, conversational tone that maintains interest
7. **Complete** - Address the full scope of what the user is asking
8. **Practical** - Include real-world applications and use cases
9. **Analytical** - Provide critical thinking and multiple perspectives
10. **Current** - Include recent developments and trends when relevant

If the question is about:
- Technical concepts: Explain with detailed examples, use cases, and applications
- Current events: Provide comprehensive context, background, and implications
- How-to questions: Give detailed step-by-step instructions with best practices
- Comparisons: Provide comprehensive analysis with pros/cons, similarities/differences
- Definitions: Give comprehensive explanations with examples, etymology, and related concepts
- Analysis: Offer deep insights, critical thinking, and multiple perspectives
- Trends: Include current developments, future directions, and implications

User: {input}

Provide a comprehensive, detailed answer with thorough analysis, practical examples, and actionable insights:
"""
                    )
                    chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
                    response = chat_chain.run(input=processed_input)
        except Exception as e:
            response = f"❌ Error: {str(e)}"
    with st.chat_message("assistant", avatar="💀"):
        if intent == "code":
            st.code(response, language="python")
        else:
            st.markdown(response)

    # Save to chat history
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.history.append((user_input, response, timestamp))
