import re
import os
import config as config

import discord
from discord.ext import commands

import torch

from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline, LLMChain, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.llms import FakeListLLM

# RUN CONFIGURATION
USE_FAKE_LLM = True
MAX_SUMMARIZATION_CHARS = 4000

# SETUP DISCORD BOT
TOKEN = config.discord_token
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(intents=intents, command_prefix="$")

# LOAD LLM MODEL
if not USE_FAKE_LLM:
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
else:
    llm = FakeListLLM(responses=['The quick brown fox jumps over the fence!'])

# SETUP LANGCHAIN LLM
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

# SETUP LANGCHAIN SUMMARIZATION
lang_sum_chain = load_summarize_chain(llm=llm, chain_type="stuff")
file_loaders = {
    'application/pdf': PyPDFLoader
}

# DEFINE BOT COMMANDS

@bot.command()
async def ask(ctx, *args):
    messsage = get_clean_msg(args)
    await ctx.reply(llm_chain.run(messsage))


def get_clean_msg(args) -> str:
    msg = ' '.join(args)
    return re.sub(r'\W ,-', '', msg)

@bot.command()
async def summarize(ctx: commands.Context, *args):
    valid_attachements = [a for a in ctx.message.attachments if a.content_type in file_loaders.keys()]
    res = {}
    for a in valid_attachements:
        await a.save(a.filename)    
        res[a.filename] = summarize_document(a)
    res_msg = "\n\n".join([f"{k}: \n\n{v}" for k,v in res.items()])
    temp_file = f'tmp_{ctx.author.id}.txt'
    with open(temp_file, 'w') as f:
        f.write(res_msg)
    with open(temp_file, 'r') as f:
        await ctx.reply(file=discord.File(f))

def summarize_document(a: discord.Attachment):
        loader = file_loaders[a.content_type](a.filename)
        docs = loader.load_and_split()
        return lang_sum_chain(docs)
            
if __name__ == "__main__":
    bot.run(TOKEN)
