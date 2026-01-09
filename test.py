import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
import cv2

from buffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork

if __name__ == "__main__":
    if not os.path.exists('tmp/td3'):
        os.makedirs('tmp/td3')

    env_name = "Cylinder"
    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        render_camera="frontview",
        has_offscreen_renderer=True,
        reward_shaping=True,
        control_freq=20
    )

    env = GymWrapper(env)

    frame = env.sim.render(
        height=480,
        width=640,
        camera_name="frontview"
    )

    video_path = "cylinder_episode.mp4"
    fps = 20
    width, height = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size, layer2_size=layer2_size,
                  batch_size=batch_size)

    n_games = 1
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"

    agent.load_model()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0.0

        while not done:
            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)
            score += reward
            observation = next_observation

            frame = env.sim.render(
                height=height,
                width=width,
                camera_name="frontview"
            )

            # Convert RGB → BGR for OpenCV
            frame_bgr = frame[:, :, ::-1]
            video.write(frame_bgr)

        video.release()
        print("Saved video to", video_path)

        print(f"Episode: {i}, Score: {score}")
