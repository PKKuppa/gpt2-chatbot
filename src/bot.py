
# %%
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import discord
import os
from dotenv import loadenv

load_dotenv()

token = os.getenv('API_KEY')
# Load the fine-tuned model and tokenizer
model_path = "../model/final"  # Change if your model is in a different directory
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate a response
def generate_response(prompt, max_length=350, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)  

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,  
            max_length=max_length,
            temperature=temperature,
            repetition_penalty=1.2,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Prevents EOS token errors
        )

 
    return response

#%% Test run
test_prompt = "User: How has your day been?\nA: "
response = generate_response(test_prompt)
print(response)
# %%

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == client.user:
        return

    # Check if the message starts with "/chat" command
    if message.content.startswith("/chat "):
        # Extract the prompt from the message
        user_prompt = message.content[len("/chat "):].strip()

        if user_prompt:
            print(f"User input: {user_prompt}")
            
            # Generate the bot's response
            user_prompt = "User: " + user_prompt + "\nA: "
            bot_response = generate_response(user_prompt)
            
            # Send the response back to the channel
            await message.channel.send(bot_response)
        else:
            await message.channel.send("Please provide a prompt after /chat. For example: `/chat How's it going?`")


client.run(token)
