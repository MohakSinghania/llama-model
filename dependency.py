import os
import uuid
import json 
import yaml
import torch
import shutil
import ollama
import secrets
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth
from typing import List
from pprint import pprint
from langchain import hub
from flask_cors import CORS
from yaml.loader import SafeLoader
from langchain_chroma import Chroma
from langchain.schema import Document
from typing_extensions import TypedDict
from streamlit import session_state as ss
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_verbose, set_debug
from flask import Flask , request , jsonify , session
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter