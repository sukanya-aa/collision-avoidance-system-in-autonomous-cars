import glob
import os
import sys

import random
import time
import numpy as np
import cv2
import math

from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.callbacks import TensorBoard
from threading import Thread
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

im_width = 640
im_height = 480
show_preview = False
seconds_per_episode = 10
replay_memory_size = 5_000
min_replay_memory_size = 1_000
mini_batch_size = 16
prediction_batch_size = 1
training_batch_size = mini_batch_size // 4
update_target_every = 5
model_name = 'Xception'
memory_fraction = 0.8
min_reward = -200
episodes = 100
discount = 0.99
epsilon = 1
epsilon_decay = 0.95
min_epsilon = 0.001
stats_aggr = 10

class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)
    
    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class carEnv:
    show_cam =show_preview
    steer_amt = 1.0

    width = im_width
    height = im_height
    actor_list = []

    front_cam = None
    collision_hist = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.5)

        self.world = self.client.get_world()

        bp_library = self.world.get_blueprint_library()
        self.model_3 = bp_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.car)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_cam.set_attribute('image_size_x', f'{self.width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.car)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.image_process(data))
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)


        col_sensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.car)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.collision_data(event))

        while self.front_cam is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()
        self.car.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_cam

    def collision_data(self, event):
        self.collision_hist.append(event)

    def image_process(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.height, self.width, 4))
        i3 = i2[:,:,:3]
        if self.show_cam:
            cv2.imshow("Image", i3)
            cv2.waitkey(1)
        self.front_cam = i3

    def step(self, action):
        '''
        0 = steer left
        1 = center
        2 = right
        '''
        if action == 0:
            self.car.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.car.apply_control(carla.VehicleControl(throttle=1.0, steer=(-1*self.steer_amt)))
        if action == 2:
            self.car.apply_control(carla.VehicleControl(throttle=1.0, steer=(1*self.steer_amt)))

        v = self.car.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:           #collision history has logs of collisions
            done = True                             # collision occured
            reward = -200                           #negative rewards
        elif kmh < 50:                              #velocity is low
            done = False                            # no collision
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + seconds_per_episode < time.time():
            done = True
        
        return self.front_cam, reward, done, None

class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False
    
    def create_model(self):
        base = Xception(weights=None, include_top=False, input_shape=(im_height, im_width, 3))

        x = base.output
        x = GlobalAveragePooling2D()(x)

        predictions = Dense(3, activation='linear')(x)
        model = Model(inputs=base.input, outputs=predictions)
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def train(self):
        if len(self.replay_memory) < min_replay_memory_size:
            return
        
        minibatch = random.sample(self.replay_memory, minibatch_size)
        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, prediction_batch_size)
        
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
             future_qs_list = self.target_model.predict(new_current_states, prediction_batch_size)
        
        x = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            x.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, 
                            shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, im_height, im_width, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)

if __name__ == '__main__':
    FPS = 60
    ep_rewards = [-200]

    random.seed(1)
    np.randome.seed(1)
    tf.set_random_seed(1)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    if not os.path.isdir('models'):
        os.makedirs('models')


    agent = DQNAgent()
    env = carEnv()  

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    agent.get_qs(np.ones((env.height, env.width, 3)))

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        env.collision_hist = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        current_state = env.reset()
        done = False
        episode_start = time.time()

        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0,3)
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        for actor in env.actor_list:
                actor.destroy()

            
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

               
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

            
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
