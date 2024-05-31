from crewai import Agent, Task, Process, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

# Instantiate Gemini AI
# To load gemini (this api is for free: https://aistudio.google.com/app/apikey)

gemini_key = 'your-secret-key' 

llm = ChatGoogleGenerativeAI(
    model="gemini-1.0-pro", verbose=True, temperature=0.1, google_api_key= gemini_key
)
# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  allow_delegation=True,
  llm = llm
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  allow_delegation=False,
  llm = llm,
)

# Research task
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  agent=researcher,
  llm = llm
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  agent=writer,
  llm = llm,
  async_execution=False,
  output_file='testcrew.md'
)

crew = Crew(
  agents=[researcher, writer],
  tasks=[research_task, write_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  verbose=2,
  share_crew=True,
  manager_llm=llm,
  max_rpm=5
)

# Starting the task execution process with enhanced feedback
result = crew.kickoff(inputs={'topic': 'AI in Hospitality'})
print("######################")

print(result)