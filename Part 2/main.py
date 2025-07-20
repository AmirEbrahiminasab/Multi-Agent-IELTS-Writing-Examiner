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


initial_state = {"original_question": """The working week should be shorter and workers should have a longer weekend.

Do you agree or disagree. """, "student_essay": """One of the hot topics these day is about how the working week needs to be shorter and the need of longer weekends. However while I believe that people need to have enough time to rest after their working period, I don't think they need longer weekends to achieve that.


Firstly, I think that there is plenty of time to rest after you get off from your work and prepare for the next day. Secondly, longer weekends will make people lazy and distant from their work. For example, when I was working at a tech company, we used to have 4 days off and when we would get back to work the connection I had with my job and colleagues didn't feel the same as I was feeling distant with my job and felt the urge to be done with the work as quickly as possible so I can go back home. Thirdly, I read a recent study that discussed the impact of cutting working hours on the economy which showed massive damage to that country's economy in the long run.


There are several ways to improve your lifestyle while not having longer weekends. Most people tend to not plan their weekends at all and that can damage the quality of their time off. Good planning will guide you to find better use of you quality time and setting the goals you want to achieve by a certain time. One of the best ways to make your job experience better is to find coworkers with the same interest as you. this was the case for me as I was having a rough few weeks at work but then ended up meeting one of my best friends who also loved soccer as much as I did and we used to discuss about our opinion on different topics which made the working experience much better for both of us.


In conclusion, I think while occasionally we need a time off from work to get into a better head space, there are other ways to tackle this problem without the need of cutting working days and making the weekends longer."""}
final_workflow = workflow_fn(State, task_response, lexical_resource, grammatical_range_and_accuracy, coherence_and_cohesion, aggregator)

final_state = final_workflow.invoke(initial_state)

display_report_fn(final_state)
