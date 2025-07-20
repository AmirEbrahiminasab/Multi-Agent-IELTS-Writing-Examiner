from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, START, END
from IPython.display import Markdown
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import requests
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
import os

from nodes import task_response, coherence_and_cohesion, lexical_resource, grammatical_range_and_accuracy, aggregator
from state import State
from workflow import workflow_fn
from display_report import display_report_fn
import base64


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


image_path = "task_image.png"
base64_image = encode_image(image_path)
image_url = f"data:image/png;base64,{base64_image}"

initial_state = {
    "image_url": image_url,
    "student_essay": """The graph below compares different proportation of people that were living in the urban from the countries of Philippines, Malaysia, Thailand and Indonesia from 1970 to 2020.



In summary, all four countries have seen an overall rose in the number of people that are living in the cities in the period shown. Philippines have had a decline between 1990 to 2010 in the percentage of the population living in the urban, but they've managed to bounce back.



Indonesia and Malaysia have had the highest rise in the proportion of people living in cities, with Malaysia having an almost 75% rate in that regard in 2020.  Also Indonesia has seen a 38% increase in the percentages of people living in the urban, going from around 12% in 1970 to 50% by the end of 2020 in that regard and having the second highest proportion by 2010.



While both Philippines and Thailand have seen an overall increase in the proportion of population living in cities, they both have had a knock along the way. Most notably, Philippines had a decline of almost 10% in their rates between the year of 1990 and 2010 and Thailand had a steady 30% rate starting from 1990 untill 2020."""
}

final_workflow = workflow_fn(State, task_response, lexical_resource, grammatical_range_and_accuracy, coherence_and_cohesion, aggregator)

final_state = final_workflow.invoke(initial_state)

display_report_fn(final_state)
