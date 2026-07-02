import os
import pickle
import time
from argparse import ArgumentParser
from multiprocessing import Process
import random
from openai import OpenAI

import zmq
# from pyarrow.serialization import deserialize, serialize
import pickle

from util.llm_client import llm_function
from util.prompt_util import prompt_function_guandan, out_parse_function_guandan
from util.llm_config import model_config


parser = ArgumentParser()
parser.add_argument('--observation_space', type=int, default=(567, ),
                    help='The YAML configuration file')
parser.add_argument('--action_space', type=int, default=(5, 216),
                    help='The YAML configuration file')
parser.add_argument('--model', type=str, default='llm',
                    help='The YAML configuration file')

class Player():
    def __init__(self, args) -> None:
        # 数据初始化
        self.args = args
        self.init_time = time.time()

        self.llm = llm_function
        self.model_name = args.model
        self.model_config = model_config[args.model]
        self.client = OpenAI(
            base_url=self.model_config['call_func']['base_url'],
            api_key=self.model_config['call_func']['api_key'],
        )
        self.prompt_function = prompt_function_guandan
        self.out_parse_function = out_parse_function_guandan
        self.request_count = 0
        self.correct_count = 0
    
    def sample(self, state) -> int:
        legal_actions = state['raw_obs']['legal_actions']
        if len(legal_actions) == 1:
            return 0

        prompt = self.prompt_function(state)
        try:
            output = self.llm(prompt, client=self.client, model_config=self.model_config)
            action = self.out_parse_function(output)
        except Exception as e:
            print(f"actor llm error: {e}; fallback to random", flush=True)
            output = ""
            action = None

        self.request_count += 1
        
        action_id = -1
        for i in range(len(legal_actions)):
            if action == legal_actions[i]:
                action_id = i
                break
        if action_id == -1:
            action_id = random.choice(range(len(legal_actions)))
        else:
            self.correct_count += 1
        # print(action)
        action_id = [action_id, output, self.request_count, self.correct_count, self.model_name]
        return action_id


def run_one_player(index, args):
    player = Player(args)

    # 初始化zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{6000+index}')
    print(f'actor{index} start')

    action_index = 0
    while True:
        # state = deserialize(socket.recv())
        state = pickle.loads(socket.recv())
        action_index = player.sample(state)
        # print(f'actor{index} do action number {action_index}')
        # socket.send(serialize(action_index).to_buffer())
        socket.send(pickle.dumps(action_index))


def main():
    # 参数传递
    args, _ = parser.parse_known_args()

    def exit_wrapper(index, *x, **kw):
        """Exit all actors on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_player(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(players):
                    if _i != index:
                        _p.terminate()

    players = []
    for i in [0, 2]:
        print(f'start{i}')
        p = Process(target=exit_wrapper, args=(i, args))
        p.start()
        time.sleep(0.5)
        players.append(p)

    for player in players:
        player.join()

if __name__ == '__main__':
    main()
