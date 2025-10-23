# Q-Learning Gridworld (Pygame)

An educational Q-learning demo in a simple gridworld rendered with Pygame. It displays parameters (alpha, gamma, epsilon), episode stats, the update equation, and a Q-value overlay.

## Controls
- Space: start/pause training
- Q: toggle Q overlay
- R: reset everything (Q-table, epsilon, episodes)
- C: clear Q-table only
- D: toggle epsilon decay
- 1/2: alpha -/+ 0.05
- 3/4: gamma -/+ 0.05
- 5/6: epsilon -/+ 0.05
- N: run one episode immediately

## Install & Run
```bash
pip install -r requirements.txt
python main.py
```