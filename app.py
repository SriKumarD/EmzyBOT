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
    -You should only answer questions related to Enzymedica products and About Enzymedica. \
    - Do not answer anything beyond these topics. If asked anything beyond Enzymedica Products or About Enzymedica Questions, Just say "Cannot answer questions that are out of context".\
    - You will be provided with a chat history . Understand the Chat history and then decided what to respond. \
    - Strictly do not answer any questions outside of  Related_Documents.\
    - If the user greets you, greet the user in return very professionaly.
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
        AIMessage(content="Hello! I'm your expert assistant for digestive health products in Mexico and Enzymedica product information. How may I assist you today?"),
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
Description_about_each_attribute = """
Product_Name: Official name of the product.
Product_URL: Direct link to the product's page on Amazon.
Rating: Average user rating of the product.
Number_of_Ratings: Total number of ratings the product has received.
Product_Price: Listed price of the product in Mexican currency.
EMI_Months: Number of months the product can be financed through Equated Monthly Installment (EMI), if applicable.
Search_For: Keywords or phrases used to search and retrieve the product listing, including:
  1. GERD supplements
  2. Traveler's Diarrhea supplements
  3. Constipation supplements
  4. Irritable Bowel Syndrome supplements
  5. Inflammatory Bowel Disease supplements
  6. Peptic Ulcers supplements
Product_URL_Hash_Key: A unique hash key generated from the Product_URL.
Reviews_URL: URL to the reviews page for the product.
Original_URL: The original URL of the product (mirroring the Product_URL).
ASIN: Amazon Standard Identification Number, a unique identifier for the product.
Product_on_Amazon_from: Date the product was first available on Amazon.
Brand: Brand name of the product.
Product_Flavor: Flavor of the product, including categories like Unflavored, Sweet Flavors, Fruit Flavors, Herbal and Spices, Citrus, Specialty Flavors, Medicinal, and Non-flavor Related.
Quantity_of_Units: Quantity of units in each package, available in grams, units, milliliters, and ounces.
Product_Form: Physical form of the product, such as Capsule, Powder, Pill, Liquid, Cream, Gel, and Drops.
Product_Color: Color of the product, including Orange, White, No Color, Beige, Yellow, Amber, Gray, Multicolor, and Pink.
Product_Dimension: Dimensions of the product.
Model_Name: Model name or number of the product.
Primary_Supplement_Type: Primary type of the supplement, such as Vitamins and Minerals, Herbal and Plant Extracts, Probiotics and Prebiotics, Omega Fatty Acids, Amino Acids and Proteins, Specialty Supplements, and Enzymes and Digestives.


 """
tools = [
    {
        "type": "function",
        "function": {
            "name": "ask_database",
            "description": "Dont call function if quetion is about Enzymedica or Enzymedica Prodcuts .Use this function only to answer user questions about Digestive  Products in Mexcain data stoed in mongodb or any question related to Digestive Concerns. Input should be a fully formed pymongo query. example query : list(collection.find())",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"""
                                -pymongo query extracting info to answer the user's question.
                                -pymongo should be written using this database schema, Description_about_each_attribute about each product and user chat_hitory:
                                -Database schema : {schema} 
                                -Description_about_each_attribute : {Description_about_each_attribute}
                                -Before  every response go through chat_history and understand user query relation with chat hisotry.
                                -Use chart history only to understand if user is asking quetion related to previous response.According to chat_history and schema write pymongo query. 
                                -The query should be returned in plain text, not in JSON. 
                                -The output strictly contain only query not anyother sybmols in it
                                -example query to find all : list(collection.find())
                                -so your task is to return one line sytax for pymongo to get data fir user query and igonre _id in every query.
                                -For searching any specific string in database use regular expressions to serach for user string and also case insenstive.
                                -you add python functions like len() to get number for particularly how many questions. Dont Do for Every Query
                                -User ask about any digestive concers check match in Concerns attribute.
                                -check query syntax before giving as syntax mistakes leads to errors.
                                -for Which questions no need to use python functions. 
                                -igonre _id for every query.
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
        return "The query is too broad and cannot be executed. Please refine your query to be more specific and try again"
    
try:
    if  (user_query is not None and user_query != "" ) :
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        flag=check_moderation_flag(user_query)
        if not flag:
            messages = [{"role": "system", "content": f"""Assume you are a Chatbot for Enzymedica company.Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous. \
                          If it is about Emzymedica dont call functions.Dont Answer anything out of tool content. If you dont have answer for user Query check any grammer error or suggets about search as a chatbot"""},
            {
                "role":"user", 
                "content": f"""user_query: {user_query},Very important Understand the chat_hitory : {st.session_state.chat_history} as user may ask questions from history."""
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
                available_functions = {
                    "ask_database": ask_database,
                }
                for tool_call in tool_calls:
                    # If true the model will return the name of the tool / function to call and the argument(s)  
                    tool_call_id = tool_call.id
                    tool_function_name = tool_call.function.name
                    function_to_call = available_functions[tool_function_name]
                    tool_query_string = json.loads(tool_call.function.arguments)['query']
                    function_response = function_to_call(
                            collection=collection,
                            query=tool_query_string,
                    )
                    function_response = str(function_response)
                    result_len=len(enc.encode(function_response))
                    if result_len > 100000:
                            function_response = ":red[Retrieval Load Exceeded. Please be more specific with your question what you are looking into.]"
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_function_name,
                            "content": f"""
                                - Summarize the result as chat bot. Here is result:{function_response}.
                                """,
                        }
                    )
                second_response = model_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                )  
                with st.chat_message("AI"):
                    st.markdown(second_response.choices[0].message.content)
                st.session_state.chat_history.append(AIMessage(content=second_response.choices[0].message.content))
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
except Exception as e:
    print(e)
    with st.chat_message("AI"):
            st.markdown(":red[We're sorry, there was a problem. Please try again.]")
            st.session_state.chat_history.append(AIMessage(content="We're sorry, there was a problem. Please try again."))
            
            
    
