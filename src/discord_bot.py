import os

import discord
from discord.ext import commands
import re

from langchain import HuggingFacePipeline, LLMChain, PromptTemplate
import torch
import os
from transformers import BitsAndBytesConfig


import src.config as config

# SETUP DISCORD BOT
TOKEN = config.discord_token
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(intents=intents, command_prefix="$")

# SETUP LANGCHAIN
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
                                            "temperature": 0, "max_length": 200,
                                            "torch_dtype": torch.float16,
                                            "low_cpu_mem_usage": True,
                                            "device_map": {'': 0}
                                        })
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


# DEFINE BOT COMMANDS

@bot.command()
async def ask(ctx, *args):
    messsage = get_clean_msg(args)
    await ctx.reply(llm_chain.run(messsage))


def get_clean_msg(args) -> str:
    msg = ' '.join(args)
    return re.sub(r'\W ,-', '', msg)


if __name__ == "__main__":
    bot.run(TOKEN)
