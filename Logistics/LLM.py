import openai
import tiktoken
import time
from openai import OpenAI

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
enc = tiktoken.encoding_for_model("gpt-4")

#model_name = 'gpt-4-32k'  # 'gpt-4' or 'gpt-3.5-turbo-0301' or 'gpt-4-32k'
openai_api_key_name = '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'  # PROMST key

def GPT_response(messages, model_name):
  if model_name in ['gpt-4-1106-preview', 'gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613']:
    #print(f'-------------------Model name: {model_name}-------------------')
    try:
      client = OpenAI(api_key = openai_api_key_name)
      response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature = 0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except:
      try:
        client = OpenAI(api_key = openai_api_key_name)
        response = client.chat.completions.create(
          model=model_name,
          messages=messages,
          temperature=0.0,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
      except:
        print(f'{model_name} Waiting 60 seconds for API query')
        time.sleep(60)
        client = OpenAI(api_key = openai_api_key_name)
        response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    return response.choices[0].message.content
  elif model_name in ["davinci-002"] and len(messages) == 2:
    prompt = messages[1]['content']
    try:
      client = OpenAI(api_key=openai_api_key_name)
      response = client.completions.create(
        model=model_name,
        prompt=prompt,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
      )
    except:
      try:
        client = OpenAI(api_key=openai_api_key_name)
        response = client.completions.create(
          model=model_name,
          prompt=prompt,
          temperature=0,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
      except:
        print(f'{model_name} Waiting 60 seconds for API query')
        time.sleep(60)
        client = OpenAI(api_key=openai_api_key_name)
        response = client.completions.create(
          model=model_name,
          prompt=prompt,
          temperature=0,
          max_tokens=256,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
    return response.choices[0].text
