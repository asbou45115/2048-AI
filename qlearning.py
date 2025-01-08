import time
import numpy as np
import pickle
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import random
import re

# Path to your Edge WebDriver executable
PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe"
service = Service(PATH)
driver = webdriver.Edge(service=service)
driver.get("https://2048game.com")

# Action chain for sending arrow key commands
actions = ActionChains(driver)

# Q-learning parameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
actions_list = ['down', 'left', 'right', 'up']
q_table = {}  # Initialize an empty Q-table

def get_game_state():
    """Fetches the current state of the game board (simplified 2D array)."""
    board = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            try:
                tile_value = driver.find_element(By.CSS_SELECTOR, f".tile-position-{i+1}-{j+1} .tile-inner").text
                board[i, j] = int(tile_value)
            except:
                board[i, j] = 0
    return board

def perform_action(action):
    """Performs the given action in the game."""
    if action == 'up':
        actions.send_keys(Keys.ARROW_UP).perform()
    elif action == 'down':
        actions.send_keys(Keys.ARROW_DOWN).perform()
    elif action == 'left':
        actions.send_keys(Keys.ARROW_LEFT).perform()
    elif action == 'right':
        actions.send_keys(Keys.ARROW_RIGHT).perform()

def choose_action(state):
    """Choose the next action based on epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        # Explore: random action
        return random.choice(actions_list)
    else:
        # Exploit: choose the best action from Q-table
        if state not in q_table:
            q_table[state] = np.zeros(len(actions_list))
        return actions_list[np.argmax(q_table[state])]

def update_q_table(prev_state, action, reward, next_state):
    """Update Q-table using the Q-learning formula."""
    if prev_state not in q_table:
        q_table[prev_state] = np.zeros(len(actions_list))
    if next_state not in q_table:
        q_table[next_state] = np.zeros(len(actions_list))
    
    action_index = actions_list.index(action)
    q_table[prev_state][action_index] = q_table[prev_state][action_index] + alpha * (
        reward + gamma * np.max(q_table[next_state]) - q_table[prev_state][action_index])

def get_score():
    """Fetches the current score from the game and returns it as an integer."""
    score_element = driver.find_element(By.CLASS_NAME, "score-container")
    score_text = score_element.text
    
    # Extract only the numeric part of the text using a regular expression
    score = re.findall(r'\d+', score_text)  # Find all groups of digits
    
    if score:
        return int(score[0])  # Convert the first found number to an integer
    else:
        return 0  # Return 0 if no number is found

def is_game_over():
    """Checks if the game is over."""
    try:
        driver.find_element(By.CLASS_NAME, "game-over")
        return True
    except:
        return False

# Main Q-learning loop
num_episodes = 1000
for episode in range(num_episodes):
    driver.find_element(By.CLASS_NAME, "restart-button").click()  # Restart game
    time.sleep(0.4)  # Wait for the game to reset
    
    prev_state = get_game_state().tobytes()  # Get initial state as bytes
    prev_score = get_score()
    
    while not is_game_over():
        action = choose_action(prev_state)  # Choose action
        perform_action(action)  # Perform action
        
        next_state = get_game_state().tobytes()  # Get next state
        new_score = get_score()  # Get new score
        
        reward = new_score - prev_score  # Reward is the score increase
        update_q_table(prev_state, action, reward, next_state)  # Update Q-table
        
        prev_state = next_state
        prev_score = new_score

    # Save Q-table after each episode
    with open("q_table_2048.pkl", "wb") as f:
        pickle.dump(q_table, f)

driver.quit()
