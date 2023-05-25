from langchain import HuggingFacePipeline, LLMChain, PromptTemplate
import torch
import os
from src import config
from transformers import BitsAndBytesConfig

os.environ['HUGGINGFACEHUB_API_TOKEN'] = config.hugging_face_hub_token

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)


repo_id = "stabilityai/stablelm-tuned-alpha-3b"
llm = HuggingFacePipeline.from_model_id(model_id=repo_id,
                                        device=0,
                                        task="text-generation",
                                        model_kwargs={
                                            "temperature": 0, "max_length": 64,
                                            "torch_dtype": torch.float16,
                                            "low_cpu_mem_usage": True,
                                            "device_map": {'': 0}
                                        })
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain('what is a duck?'))
