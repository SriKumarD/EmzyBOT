import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from pymongo import MongoClient
import os
import json
from openai import OpenAI
from pinecone import Pinecone
import tiktoken
import json

_ = load_dotenv()
os.environ.get("OPENAI_API_KEY")
pinecone_api_key= os.environ.get("PINECONE_API_KEY")
MongoClient_URl = os.environ.get('MongoClient_URl')
model_client = OpenAI()
# MongoDB connection details
mongo_uri = MongoClient_URl
client = MongoClient(mongo_uri)
pc = Pinecone(api_key=pinecone_api_key)
# Create or connect to the collection
db = client['EnzymedicaDB']
collection = db['Products']
enc = tiktoken.encoding_for_model("gpt-4o")


index_name = 'langchain-index-emzym'
index = pc.Index(index_name)
embed_model = "text-embedding-3-small"


load_dotenv()

# app config
st.set_page_config(page_title="Enzymedica Digestive Advisor", page_icon="👨‍🔬")
st.title(" Enzymedica Digestive Advisor")
st.caption("Your expert assistant for digestive health products in Mexico and Enzymedica product information.")
def get_response(user_query, chat_history,documets):
    template = """
    ("system", "You are a marketing Advisor for Enzymedica, which is a digestive enzymes & health supplements company. 
    -You should only answer questions related to Enzymedica products and gut health or Product Questions. \
    - Do not answer anything beyond these topics. If asked anything beyond \
    - Enzymedica Products or Gut health or Problem related Questions, Just say "Cannot answer questions that are out of context".\
    - You will be provided with a chat history related documents and a user question. \
    - Strictly do not answer any questions outside of Enzymedica or related documents.\
    - If the user greets you, greet the user in return
            ")
    ("human", "Answer the following questions considering the history of the conversation and documents related to Question:
    Chat history: {chat_history} \
    User question: {user_question} \
    Related_Documents : {documets} \
    ")
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model='gpt-4o')
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "documets" : documets
    })

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

   
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

def check_moderation_flag(expression):
    moderation_response = model_client.moderations.create(input=expression)
    flagged = moderation_response.results[0].flagged
    return flagged


user_query = st.chat_input("Type your message here...")


def get_collection_schema(db_name):
    db = client[db_name]
    collections = db.list_collection_names()
    
    schema = {}
    
    for collection_name in collections:
        collection = db[collection_name]
        documents = collection.find().limit(100)  # Limit to 100 documents to infer schema
        df = pd.DataFrame(list(documents))
        
        schema[collection_name] = {}
        
        for column in df.columns:
            dtype = df[column].apply(lambda x: type(x).__name__).unique()
            schema[collection_name][column] = list(dtype)
    
    return schema

db_name = "EnzymedicaDB"
schema = get_collection_schema(db_name)

tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Dont call if quetion is about Enzymedica .Use this function only to answer user questions about Digestive  Products in Mexcain data stoed in mongodb. Input should be a fully formed pymongo query. example query : list(collection.find())",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                -pymongo query extracting info to answer the user's question.
                                -pymongo should be written using this database schema and chat_hitory:
                                database schema : {schema} and chat_hitory : {st.session_state.chat_history}
                                - Use chart history only to understand if user is asking quetion related to previous response.According to chat_history and schema write pymongo query. 
                                -The query should be returned in plain text, not in JSON. 
                                -The output strictly contain only query not anyother sybmols in it
                                -example query to find all : list(collection.find())
                                -so your task is to return one line sytax for pymongo to get data fir user query and igonre _id in every query.
                                -For searching any specific string in database use Regular expressions and also case insenstive.
                                -you add python functions like len() to get number for particularly how many questions. Dont Do for Every Query
                                -User ask about any digestive concers check match in Concerns attribute.
                                -check query syntax before giving as syntax mistakes leads to errors.
                                -for Which questions no need to use python functions. 
                                """,
                    }
                },
                "required": ["query"],
            },
        }
    },
    
]


def ask_database(collection,query):
    """Function to query MongoDB database with a provided MongoDB query."""
    try:
        results = eval(query)
        if results:
            return results
        else:
            return "No documents match the query."
    except Exception as e:
        return query


if  (user_query is not None and user_query != "" ) :
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    flag=check_moderation_flag(user_query)
    if not flag:
        messages = [{
            "role":"user", 
            "content": user_query
        }]

        response = model_client.chat.completions.create(
            model='gpt-4o', 
            messages=messages, 
            tools= tools, 
            tool_choice="auto"
        )

        # Append the message to messages list
        response_message = response.choices[0].message 
        messages.append(response_message)
        tool_calls = response_message.tool_calls

        if tool_calls:
            # If true the model will return the name of the tool / function to call and the argument(s)  
            tool_call_id = tool_calls[0].id
            tool_function_name = tool_calls[0].function.name
            tool_query_string = json.loads(tool_calls[0].function.arguments)['query']
            
            # Step 3: Call the function and retrieve results. Append the results to the messages list.      
            if tool_function_name == 'ask_database':
                results = ask_database(collection,tool_query_string)
                result_len=len(enc.encode(str(results)))
                if result_len > 100000:
                    with st.chat_message("AI"):
                        st.markdown(":red[Retrieval Load Exceeded. Please be more specific with your question what you are looking into.]")
                    st.session_state.chat_history.append(AIMessage(content="Retrieval Load Exceeded. Please be more specific with your question of what you are looking into."))
                else:
                    messages.append({
                        "role":"tool", 
                        "tool_call_id":tool_call_id, 
                        "name": tool_function_name, 
                        "content":f"""
                        - Summarize the result as chat bot. Here is result:{results}.\
                        - check is result:{results} empty. If it is empty Just say "Please be more specific with your question and stay within the context of the discussion."
                        - If result are empty Just say "Please be more specific with your question and stay within the context of the discussion."
                        ***Strictly check If user_query: {user_query} is something not about the  country Mexcio ,Just say "Please be more specific with your question and stay within the context of the discussion."***
                        """
                    })
                    
                    # Step 4: Invoke the chat completions API with the function response appended to the messages list
                    # Note that messages with role 'tool' must be a response to a preceding message with 'tool_calls'
                    model_response_with_function_call = model_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                    )  # get a new response from the model where it can see the function response
                    
                    with st.chat_message("AI"):
                        st.markdown(model_response_with_function_call.choices[0].message.content)
                    st.session_state.chat_history.append(AIMessage(content=model_response_with_function_call.choices[0].message.content))
            else: 
                with st.chat_message("AI"):
                    res = model_client.embeddings.create(
                            input=[user_query],
                            model=embed_model
                        )
                    # retrieve from Pinecone
                    xq = res.data[0].embedding

                    # get relevant contexts (including the questions)
                    res = index.query(vector=xq, top_k=5, include_metadata=True)
                    # Extract 'text' from each metadata dictionary within the matches list
                    product_texts = [match['metadata']['text'] for match in res['matches']]

                    documets=[]
                    for text in product_texts:
                        documets.append(text)
                    response = st.write_stream(get_response(user_query,st.session_state.chat_history,documets))
                st.session_state.chat_history.append(AIMessage(content=response))
        else:
            with st.chat_message("AI"):
                    res = model_client.embeddings.create(
                            input=[user_query],
                            model=embed_model
                        )
                    # retrieve from Pinecone
                    xq = res.data[0].embedding

                    # get relevant contexts (including the questions)
                    res = index.query(vector=xq, top_k=5, include_metadata=True)
                    # Extract 'text' from each metadata dictionary within the matches list
                    product_texts = [match['metadata']['text'] for match in res['matches']]

                    documets=[]
                    for text in product_texts:
                        documets.append(text)
                    response = st.write_stream(get_response(user_query,st.session_state.chat_history,documets))
            st.session_state.chat_history.append(AIMessage(content=response))
    else:
        with st.chat_message("AI"):
            st.markdown(":red[We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again.]")
        st.session_state.chat_history.append(AIMessage(content=":red[We're sorry, but your input has been flagged as inappropriate. Please rephrase your input and try again.]"))
 
            
            
    
