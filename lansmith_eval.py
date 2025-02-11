from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama

# ✅ Initialize LangSmith Client
import os
from langsmith import wrappers, Client
from apikeys import LangSmith_API  # Import API key

# ✅ Set LangSmith API Key
os.environ["LANGCHAIN_API_KEY"] = LangSmith_API

# ✅ Initialize LangSmith Client
client = Client()


# client = Client()

# ✅ Initialize Ollama Model (Llama 2)
ollama_client = Ollama(model="llama2", base_url="http://127.0.0.1:11500")

# ✅ Define Evaluation Dataset
examples = [
    ("Which country is Mount Kilimanjaro located in?", "Mount Kilimanjaro is located in Tanzania."),
    ("What is Earth's lowest point?", "Earth's lowest point is The Dead Sea."),
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

# ✅ Programmatically create a dataset in LangSmith
dataset = client.create_dataset(
    dataset_name="Sample dataset",
    description="A sample dataset in LangSmith using Ollama."
)

# ✅ Add examples to the dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)

# ✅ Define the target function using Ollama instead of OpenAI
def target(inputs: dict) -> dict:
    response = ollama_client.invoke(inputs["question"])
    return {"response": response.strip()}

# ✅ Define Evaluation Instructions
instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
- False: No conceptual match and similarity
- True: Most or full conceptual match and similarity
- Key criteria: Concept should match, not exact wording.
"""

# ✅ Define Output Schema for the LLM Judge
class Grade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the response is accurate relative to the reference answer")

# ✅ Define LLM Judge That Uses Ollama for Evaluation
def accuracy(outputs: dict, reference_outputs: dict) -> bool:
    response = ollama_client.invoke(
        f"""System Instructions: {instructions}
        
        Ground Truth Answer: {reference_outputs["answer"]}
        Student's Answer: {outputs["response"]}
        
        Evaluate the response and return True if it is conceptually correct, else False.
        """
    )
    return response.lower() == "true"

# ✅ Run Evaluation in LangSmith
experiment_results = client.run_on_dataset(
    dataset_name="Sample dataset",  # Match dataset name
    llm_or_chain_factory=target,  # Your target function (Ollama inference)
    evaluators=["qa"],  # Use LangSmith's built-in evaluation
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)

print("Evaluation completed! Check results in LangSmith.")
