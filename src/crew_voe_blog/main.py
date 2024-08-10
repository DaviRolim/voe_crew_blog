#!/usr/bin/env python
import sys
from crew_voe_blog.crew import CrewVoeBlogCrew
from dotenv import load_dotenv

load_dotenv()

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding necessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        "topic_of_interest": "Dicas para voos de longa duração",
        "current_year": 2024
    }
    CrewVoeBlogCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic_of_interest": "Dicas para voos de longa duração",
        "current_year": 2024
    }
    try:
        CrewVoeBlogCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CrewVoeBlogCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
