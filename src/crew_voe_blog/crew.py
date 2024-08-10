from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from langchain.agents import Tool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# llm =  ChatOpenAI(model_name="gpt-4o-mini")
llm = Ollama(
    model = "llama3.1",
    base_url = "http://localhost:11434")
dalle = DallEAPIWrapper(n=1, size="512x512")

dalle_tool = Tool(
    name="DALL-E API",
    func=dalle.run,
    description="useful for generating images. The input to this should be a detailed prompt to generate an image in English.",
)

@CrewBase
class CrewVoeBlogCrew():
	"""CrewVoeBlog crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def flight_news_reporter(self) -> Agent:
		return Agent(
			config=self.agents_config['flight_news_reporter'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

	@agent
	def travel_tips_curator(self) -> Agent:
		return Agent(
			config=self.agents_config['travel_tips_curator'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			llm=llm,
			allow_delegation=False,
			verbose=True
		)

	@agent
	def step_by_step_guide_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['step_by_step_guide_creator'],
			tools=[SerperDevTool(), ScrapeWebsiteTool()],
			llm=llm,
			allow_delegation=False,
			verbose=True
		)

	@agent
	def travel_content_creator(self) -> Agent:
		return Agent(
			config=self.agents_config['travel_content_creator'],
			tools=[dalle_tool],
			llm=llm,
			allow_delegation=False,
			verbose=True
		)


	@task
	def news_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['news_research_task'],
			agent=self.flight_news_reporter()
		)

	@task
	def travel_tips_curation_task(self) -> Task:
		return Task(
			config=self.tasks_config['travel_tips_curation_task'],
			agent=self.travel_tips_curator(),
		)
	@task
	def how_to_guide_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['how_to_guide_creation_task'],
			agent=self.step_by_step_guide_creator()
		)
	@task
	def blog_post_creation_task(self) -> Task:
		return Task(
			config=self.tasks_config['blog_post_creation_task'],
			agent=self.travel_content_creator(),
			context=[self.news_research_task(), self.travel_tips_curation_task(), self.how_to_guide_creation_task()],
			output_file='report.md'
		)




	@crew
	def crew(self) -> Crew:
		"""Creates the CrewVoeBlog crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=2,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)