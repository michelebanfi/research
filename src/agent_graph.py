import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Union, List, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START

from src.ai_engine import AIEngine
from src.database import DatabaseClient
from src.config import Config

# --- State Definition ---
class AgentState(TypedDict):
    """The state of the agent's reasoning graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    documents: List[Dict[str, Any]]
    generation: str
    relevance_score: str  # "yes" or "no"
    retry_count: int  # Track transform → retrieve cycles to prevent infinite loops

# --- Node Components ---

class GraphNodes:
    def __init__(self, ai_engine: AIEngine, db: DatabaseClient, project_id: str):
        self.ai = ai_engine
        self.db = db
        self.project_id = project_id
        
        # Initialize LangChain ChatModel wrapper around our OpenRouter config
        # We use the key from config/env
        self.llm = ChatOpenAI(
            model=Config.OPENROUTER_MODEL,
            openai_api_key=Config.OPENROUTER_KEY, 
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0
        )

    async def retrieve(self, state: AgentState):
        """
        Retrieve documents based on the latest user query.
        """
        print("---RETRIEVE---")
        query = state["user_query"]
        
        # Use our existing vector search logic
        query_embedding = self.ai.generate_embedding(query)
        documents = []
        if query_embedding:
            documents = self.db.search_vectors(
                query_embedding, 
                match_threshold=0.3, 
                project_id=self.project_id, 
                match_count=5
            )
            
        return {"documents": documents}

    async def grade_documents(self, state: AgentState):
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        print("---CHECK RELEVANCE---")
        question = state["user_query"]
        documents = state["documents"]
        
        # LLM grader
        system_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        filtered_docs = []
        has_relevant = "no"
        
        for d in documents:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Retrieved document: \n\n {d['content']} \n\n User question: {question}")
            ]
            
            response = await self.llm.ainvoke(messages)
            score = response.content.lower()
            
            if "yes" in score:
                filtered_docs.append(d)
                has_relevant = "yes"
            else:
                continue
                
        return {"documents": filtered_docs, "relevance_score": has_relevant}

    async def generate(self, state: AgentState):
        """
        Generate answer.
        """
        print("---GENERATE---")
        question = state["user_query"]
        documents = state["documents"]
        
        context = "\n\n".join([d['content'] for d in documents])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            ("human", "Question: {question} \n\n Context: {context} \n\n Answer:")
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({"question": question, "context": context})
        
        return {"generation": response.content, "messages": [response]}

    async def transform_query(self, state: AgentState):
        """
        Transform the query to produce a better question.
        """
        print("---TRANSFORM QUERY---")
        question = state["user_query"]
        
        msg = [
            SystemMessage(content="You are generating a question that is well optimized for retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."),
            HumanMessage(content=f"Initial question: {question} \n Formulate an improved question:")
        ]
        
        response = await self.llm.ainvoke(msg)
        better_question = response.content
        
        return {"user_query": better_question, "retry_count": state.get("retry_count", 0) + 1}

# --- Graph Builder ---

def build_graph(ai_engine: AIEngine, db: DatabaseClient, project_id: str):
    nodes = GraphNodes(ai_engine, db, project_id)
    
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)

    # Build the graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    MAX_RETRIES = 2  # Prevent infinite retrieve → grade → transform cycles
    
    def decide_to_generate(state):
        if state.get("retry_count", 0) >= MAX_RETRIES:
            print(f"---MAX RETRIES ({MAX_RETRIES}) REACHED, FORCING GENERATION---")
            return "generate"
        if state["relevance_score"] == "yes":
            return "generate"
        else:
            return "transform_query"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )
    
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app

# --- Testing / Run Wrapper ---
async def run_agent_graph(user_query: str, ai_engine, db, project_id: str):
    app = build_graph(ai_engine, db, project_id)
    
    inputs = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "retry_count": 0
    }
    
    # We will just return the final state's generation
    final_state = await app.ainvoke(inputs)
    return final_state["generation"]
