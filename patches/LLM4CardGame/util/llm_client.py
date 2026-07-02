import os, json
import requests
import re
from tenacity import retry, stop_after_attempt, wait_fixed

temperature = os.getenv('API_TEMP', 0)

def openai_api_function(prompt, history=None, system=None, client=None, model_config=None):
    message = []
    if system:
        message.append({
            "role": "system",
            "content": system
        })
    
    if history:
        for chat in history:
            message.append({
                "role": "user",
                "content": chat[0]
            })
            message.append({
                "role": "assistant",
                "content": chat[1]
            })
    
    message.append({
        "role": "user",
        "content": prompt
    })

    # print(prompt)
    # model_type = "THUDM/glm-4-9b-chat"
    resp = client.with_options(max_retries=5).chat.completions.create(
        messages=message,
        model=model_config['model_type'],
        temperature=temperature,
        # do_sample=model_config['do_sample'],
        stream=False,
        max_tokens=256
    )
    output = resp.choices[0].message.content
    # print(output)

    # # import pdb
    # # pdb.set_trace()
    # with open(os.path.join(LOG_PATH, model_type), 'a', encoding="utf-8") as fout:
    #     j_data = {'model': model_type, 'input': prompt, 'output': output}
    #     fout.write(json.dumps(j_data))
    #     fout.write('\n')

    return output

class ModelCallError(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry_error_cls=ModelCallError)
def llm_function(prompt, history=None, system=None, client=None, model_config=None):
    try:
        output = openai_api_function(prompt, history, system, client, model_config)
    except Exception as e:
            raise ModelCallError(f"Error calling model: {str(e)}")
    return output