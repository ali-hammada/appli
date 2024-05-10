import random
import gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from time import datetime
import sqlite3
import logging
from collections import deque


logging.basicConfig(level=logging.INFO)

class Database:
    def __init__(self, db_file):
        try:
            self.db_file = db_file
            self.connection = sqlite3.connect(db_file)
            self.cursor = self.connection.cursor()
            self.activities = []  
            logging.info("Connected to database successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")

    def close_connection(self):
        try:
            if self.connection:
                self.connection.close()
                logging.info("Database connection closed.")
        except sqlite3.Error as e:
            logging.error(f"Error closing database connection: {e}")

    def get_ml_prediction(self):
        try:
            # Execute SQL query to get ML prediction from the database
            self.cursor.execute("SELECT prediction FROM ml_predictions ORDER BY id DESC LIMIT 1")
            prediction = self.cursor.fetchone()[0]
            return prediction
        except sqlite3.Error as e:
            logging.error(f"Error getting ML prediction from database: {e}")
            return None
    
    def get_initial_state(self):
        try:
            # Execute SQL query to get the initial state from the database
            self.cursor.execute("SELECT initial_state FROM initial_states WHERE date = ?", (datetime.date.today(),))
            initial_state = self.cursor.fetchone()[0]
            return initial_state
        except sqlite3.Error as e:
            logging.error(f"Error getting initial state from database: {e}")
            return None

    def get_user_feedback(self, predicted_activity):
        try: 
            # Execute SQL query to get user feedback for the predicted activity from the database
            self.cursor.execute("SELECT feedback FROM user_feedback WHERE activity = ?", (predicted_activity,))
            feedback = self.cursor.fetchone()[0] 
            return feedback
        except sqlite3.Error as e:
            logging.error(f"Error getting user feedback from database: {e}")
            return None  

    def update_state_with_feedback(self, user_feedback, current_state):
        try:
            # Update the state in the database with the user feedback
            # This could involve updating a table or some other operation depending on your application
            pass
        except sqlite3.Error as e:
            logging.error(f"Error updating state with feedback in database: {e}")

    def check_if_day_ends(self):
        try:
            # Check if the day ends based on some condition in the database
            # For example, you might check if the current time exceeds a certain threshold
            return False
        except sqlite3.Error as e:
            logging.error(f"Error checking if day ends in database: {e}")
            return False 


class DatabaseInterface:
    def __init__(self, database):
        self.database = database
        
    def get_initial_state(self):
        initial_state = self.database.get_initial_state()
        return initial_state

    def get_user_feedback(self, predicted_activity):
        user_feedback = self.database.get_user_feedback(predicted_activity)
        return user_feedback

    def update_state_with_feedback(self, user_feedback, current_state):
        updated_state = self.database.update_state_with_feedback(user_feedback, current_state)
        return updated_state

    def check_if_day_ends(self):
        done = self.database.check_if_day_ends()
        return done


class DailyActivityEnv(gym.Env):
    def __init__(self, db_interface, replay_buffer, simulation_speed=1.0):
        super(DailyActivityEnv, self).__init__()
        self.db_interface = db_interface
        self.replay_buffer = replay_buffer
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1440, shape=(1,), dtype=np.int32)
        self.simulation_speed = simulation_speed

    def reset(self):
        self.current_state = self.db_interface.get_initial_state()
        return np.array([self.current_state])

    def step(self, action):
        predicted_activity = self.predict_activity(self.current_state)
        user_feedback = self.db_interface.get_user_feedback(predicted_activity)
        reward = self.calculate_reward(user_feedback, action)
        next_state = self.db_interface.update_state_with_feedback(user_feedback, self.current_state)
        done = self.db_interface.check_if_day_ends()
        self.replay_buffer.add((self.current_state, action, reward, next_state, done))
        self.current_state = next_state
        logging.info(f"Action: {action}, User Feedback: {user_feedback}, Reward: {reward}, Next State: {next_state}, Done: {done}")
        return np.array([self.current_state]), reward, done, {}

    def predict_activity(self, state):
        predicted_activity = np.random.randint(low=0, high=len(self.db_interface.database.activities))
        return predicted_activity

    def calculate_reward(self, user_feedback, action):
        if user_feedback == 'Yes' and action == 1:
            return 1
        elif user_feedback == 'No' and action == 0:
            return -1
        else:
            return 0


class CustomCNN(nn.Module):
    def __init__(self, observation_space):
        super(CustomCNN, self).__init__()
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# Setup logging
logging.basicConfig(level=logging.INFO)
database = Database("database.db")
db_interface = DatabaseInterface(database)
replay_buffer = ExperienceReplayBuffer(capacity=10000)
env = DailyActivityEnv(db_interface, replay_buffer, simulation_speed=1.0)
policy_kwargs = dict(features_extractor_class=CustomCNN)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=0.001, verbose=1)
model.learn(total_timesteps=50000)

database.close_connection()
