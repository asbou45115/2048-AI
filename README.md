# 2048 Q-Learning Agent with Parallel Processing

This project implements a reinforcement learning agent that plays the 2048 game using the Q-learning algorithm. The agent interacts with the 2048 game via Selenium to perform actions and observe the game state. The Q-table is used to learn the best strategies through trial and error. To improve efficiency, multiple agents are run in parallel using Python's multiprocessing module.

## Features

- **Q-learning**: The agent learns how to play the game through reinforcement learning.
- **Parallel Processing**: Multiple agents are run concurrently to speed up the training process.
- **Selenium Integration**: The agent interacts with the 2048 game through Selenium WebDriver.
- **Q-table Persistence**: The Q-table is saved after every episode, allowing you to resume training.

## Requirements

- Python 3.7 or higher
- Microsoft Edge Browser
- Microsoft Edge WebDriver (compatible version)
- Python packages:
  - `selenium`
  - `numpy`
  - `pickle`

## Installation

1. **Install Python Dependencies**:

    Run the following command to install the required packages:

    ```bash
    pip install selenium numpy
    ```

2. **Set Up WebDriver**:

    - Download the Microsoft Edge WebDriver that matches your version of Edge from [here](https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/).
    - Extract and place it in a directory of your choice (e.g., `C:\Program Files (x86)\Microsoft\Edge\Application\`).

3. **Clone the Repository**:

    Clone this repository or download the project files manually.

4. **Edit the WebDriver Path**:

    Update the `PATH` variable in the script to point to the correct location of your `msedgedriver.exe`:

    ```python
    PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedgedriver.exe"
    ```

## Usage

To run the training script with parallel processing, execute the script from the terminal:

```bash
python3 qlearning.py
```

## Q-Learning Algorithm

- Learning Rate (α): The rate at which the agent updates the Q-values for state-action pairs. Set to 0.1 by default.
- Discount Factor (γ): The factor that weighs future rewards versus immediate rewards. Set to 0.9 by default.
- Exploration Rate (ε): Controls how often the agent explores new actions instead of exploiting the known best actions. Set to 0.1.

### Formula: 
```bash
Q(s, a) ← Q(s, a) + α [r + γ max(Q(s', a')) - Q(s, a)]
```
Where:

- Q(s, a) is the Q-value for the current state s and action a.
- r is the immediate reward.
- s' is the next state after taking action a.
- α is the learning rate.
- γ is the discount factor.

## Customization

- Number of Agents: Change the num_agents variable to control how many browser instances (agents) run in parallel.
- Number of Episodes: Modify num_episodes_per_agent to control how many episodes each agent plays during training.

The current number is 4 agents with 10 episodes each for a combined 40, this was due to constraints of the local pc and efficency and so can be further improved with more iterations and tweaks to the reward function. The basis of this project was solely for the education of q-learning.
