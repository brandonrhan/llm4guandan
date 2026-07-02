import os

port = os.getenv('API_PORT')
port = port if port else 8555
port2 = os.getenv('API_PORT2')
port2 = port2 if port2 else 8556

model_provider = {
    'local': {
        'base_url': f"http://0.0.0.0:{port}/v1",
        'api_key': 'local',
        'num_workers': 5,
    },
    'local2': {
        'base_url': f"http://0.0.0.0:{port2}/v1",
        'api_key': 'local2',
        'num_workers': 5,
    },
    'openai': {
        'base_url': "openai base url",
        'api_key': 'your openai api key',
        'num_workers': 1,
    },
    'glm': {
        'base_url': "glm base url",
        'api_key': 'your glm api key',
        'num_workers': 1,
    }
}

model_config = {
    'llm': {
        'name': 'llm',
        'model_type': 'guandan',
        'do_sample': False,
        'call_func': model_provider['local'],
    },
    'llm2': {
        'name': 'llm2',
        'model_type': 'guandan',
        'do_sample': False,
        'call_func': model_provider['local2'],
    },
    'gpt4o': {
        'name': 'gpt4o',
        'model_type': 'gpt-4o-2024-05-13',
        'do_sample': False,
        'call_func': model_provider['openai'],
    },
    'gpt4om': {
        'name': 'gpt4om',
        'model_type': 'gpt-4o-mini-2024-07-18',
        'do_sample': False,
        'call_func': model_provider['openai'],
    },
    'glm4-plus': {
        'name': 'glm4-plus',
        'model_type': 'glm-4-plus',
        'do_sample': False,
        'call_func': model_provider['glm'],
    },
    'glm4-air': {
        'name': 'glm4-air',
        'model_type': 'glm-4-air',
        'do_sample': False,
        'call_func': model_provider['glm'],
    }
}

