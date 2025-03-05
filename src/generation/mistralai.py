import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import xml.etree.ElementTree as ET
from xml.dom import minidom

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_id = 'mistralai/Mistral-7B-Instruct-v0.3'

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token='your_key',
)

tokenizer.chat_template = """
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] | trim + '\n\n' -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {% set system_message = '' %}
{%- endif -%}
{{ bos_token + system_message}}
{%- for message in messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {%- endif -%}
    {%- if message['role'] == 'user' -%}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {%- elif message['role'] == 'assistant' -%}
        {{ ' ' + message['content'] | trim + eos_token }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{-''-}}
{%- endif -%}
"""

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token='your_key',
    #torch_dtype=torch.float16,
    device_map='cuda',
)

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
)

messages = [
    {
        'role': 'system',
        'content': 'You are a student with an A1 proficiency in English.' \
            'Your class teacher will ask you a question.' \
            'You should answer this question with the minimum of 30 words.',
    },
    {
        'role': 'user', 
        'content': 'What are your daily habits? What time do you get up, etc.?'
    },
]

df = pd.read_csv('prompts.csv')

responses = []
for index, row in df.iterrows():
    prompt_id = row['id']
    prompt_text = row['prompts']
    
    for _ in range(5):
        chat = pipe(
            prompt_text,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        output = chat[0]['generated_text'].replace(prompt_text, '').strip()
        print(output)
        responses.append((prompt_id, prompt_text, output))

for i, (prompt_id, prompt_text, output) in enumerate(responses):
    root = ET.Element('root')
    questions = ET.SubElement(root, 'questions')
    question = ET.SubElement(questions, 'question')
    question.text = prompt_text

    answers = ET.SubElement(root, 'answers')
    paragraphs = output.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            answer = ET.SubElement(answers, 'answer')
            answer.text = paragraph.strip()

    rough_str = ET.tostring(root, 'utf-8')
    reparsed = minidom.parseString(rough_str)
    pretty_str = reparsed.toprettyxml(indent='    ')

    with open(f'mistralai-corpus/{i}.xml', 'w') as file:
        file.write(pretty_str)
