# -*- coding: utf-8 -*-

import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from cnn_dqn_agent import CnnDqnAgent
import msgpack
import io
from PIL import Image
from PIL import ImageOps
import threading
import numpy as np
import datetime
import os

parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help='websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help='server ip')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help='reward log file name')
args = parser.parse_args()


class Root(object):
    @cherrypy.expose
    def index(self):
        return 'some HTML with a websocket javascript connection'

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler


class AgentServer(WebSocket):
    agent = CnnDqnAgent()
    agent_initialized = False
    cycle_counter = 0
    thread_event = threading.Event()
    log_file = args.log_file
    reward_sum = 0
    depth_image_dim = 32 * 32
    depth_image_count = 1
    now = datetime.datetime.now()
    otherStyleTime = now.strftime("%Y-%m-%d %H_%M_%S")

    cur_dir = 'C:\Users\hosilab\Desktop\ls\ls\python-agent'
    folder_name = 'RGB'
    directory = cur_dir + '\\' + folder_name
    if os.path.isdir(cur_dir)and os.path.exists(directory)is False:
        os.makedirs(directory)
        #os.mkdir(os.path.join(cur_dir, folder_name))
        #folder_name = folder_name

        #+ otherStyleTime[-1:-9:-1][::-1]
    def send_action(self, action):
        dat = msgpack.packb({"command": str(action)})
        self.send(dat, binary=True)

    def received_message(self, m):
        payload = m.data
        dat = msgpack.unpackb(payload)

        image = []
        for i in xrange(self.depth_image_count):
            image_ = Image.open(io.BytesIO(bytearray(dat['image'][i])))
            #image_.save("./RGB/" + "img_" + str(self.cycle_counter) + ".png")
            image.append(image_)
        #depth = []
        # for i in xrange(self.depth_image_count):
        #     d = (Image.open(io.BytesIO(bytearray(dat['depth'][i]))))
        #     depth.append(np.array(ImageOps.grayscale(d)).reshape(self.depth_image_dim))

        observation = {"image": image}#, "depth": depth}
        #print observation["image"][0], "observation" #observation["image"]=
        # [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=227x227 at 0x7F60F28>]
        #[<PIL.PngImagePlugin.PngImageFile image mode=RGB size=227x227 at 0x7E61FD0>]
        #reward = dat['reward']#np.array(dat['reward'], dtype=np.float32)#dat['reward']#<type 'float'>#
        reward = np.asanyarray(dat['reward'], dtype=np.float32)#-----------------------------------
        #print(reward.ndim, 'Reward!!length')#error#------------------------------------------------
        end_episode = dat['endEpisode']

        if not self.agent_initialized:
            self.agent_initialized = True
            print("initializing agent...")
            self.agent.agent_init(
                use_gpu=args.gpu)#,
                #depth_image_dim=self.depth_image_dim * self.depth_image_count)

            action = self.agent.agent_start(observation)#开始的第一个动作是白送的
           # # if action != self.agent.actions.index(2):
           #      reward = -0.3
           #  else:
           #      reward = 0.1
           #  #print reward, "!!!!!!!!!!!!!"
            self.send_action(action)
            with open(self.log_file, 'w') as the_file:
                the_file.write('cycle, episode_reward_sum \n')
        else:#如果agent启动观测环境和动作给了，大如果
            self.thread_event.wait()
            self.cycle_counter += 1
            self.reward_sum += reward#此处reward要改#---------------------------

            if end_episode:#如果在大如果下 结束这一回合，通过开始的state_获得动作和q值，然后下个状态就是开始的状态
                self.agent.agent_end(reward)
                action = self.agent.agent_start(observation)  # TODO# return return_action
                                                 # #action, q_now = self.q_net.e_greedy(state_, self.epsilon)-75
                # if action != self.agent.actions.index(2):
                #     reward -= 0.3
                self.send_action(action)
                with open(self.log_file, 'a') as the_file:
                    the_file.write(str(self.cycle_counter) +
                                   ',' + str(self.reward_sum) + '\n')
                self.reward_sum = 0
            else:#如果在大如果下没有结束这个回合
                action, eps, q_now, obs_array = self.agent.agent_step(reward, observation)
                if action != self.agent.actions.index(2)and reward != 1.:
                    reward -= 0.1 #先执行这个
                self.send_action(action)
                self.agent.agent_step_update(reward, action, eps, q_now, obs_array)
                #通过cnn——agentupdata self.q_net.stock_experience和 self.q_net.experience_replay
        self.thread_event.set()

cherrypy.config.update({'server.socket_host': args.ip,
                        'server.socket_port': args.port})
WebSocketPlugin(cherrypy.engine).subscribe()
cherrypy.tools.websocket = WebSocketTool()
cherrypy.config.update({'engine.autoreload.on': False})
config = {'/ws': {'tools.websocket.on': True,
                  'tools.websocket.handler_cls': AgentServer}}
cherrypy.quickstart(Root(), '/', config)

