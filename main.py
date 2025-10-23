import pygame, sys, numpy as np, random

# ---- Config ----
GRID_W, GRID_H = 5, 5
CELL = 110
SIDEBAR = 540
BOTTOM = 320
W, H = 0, 0
# Rewards per spec
STEP_REWARD = -1
HOLE_REWARD = -10
GOAL_REWARD = 100
BLOCK_PENALTY = -1

# ---- Env layout ----
walls = {(2,1),(2,2),(2,3)}
holes = set()
start = (0,0)
goal = (4,4)

# ---- Colors ----
BLACK=(15,15,15); WHITE=(230,230,230); GREY=(80,80,80)
GRID=(45,45,45); WALL=(40,40,40); HOLE=(180,50,50); GOAL=(50,180,80)
AGENT=(80,160,240); TEXT=(220,220,220)

pygame.init()
screen = pygame.display.set_mode((1280, 900), pygame.RESIZABLE)
W, H = screen.get_size()

# recompute layout dynamically based on window size
def compute_layout():
    global W, H, CELL, SIDEBAR, BOTTOM
    W, H = screen.get_size()
    # allocate portions for sidebar and bottom, ensuring minimums
    SIDEBAR = max(360, int(W * 0.35))
    BOTTOM  = max(220, int(H * 0.25))
    # fit CELL so grid uses remaining space cleanly
    CELL = max(40, min((W - SIDEBAR)//GRID_W, (H - BOTTOM)//GRID_H))
    # snap sidebar/bottom to the leftover so panels align perfectly
    SIDEBAR = W - GRID_W*CELL
    BOTTOM  = H - GRID_H*CELL

compute_layout()
pygame.display.set_caption("Q-Learning Gridworld (Educational Demo)")
font = pygame.font.SysFont(None, 24)
small = pygame.font.SysFont(None, 18)
clock = pygame.time.Clock()

# ---- Q-Learning ----
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
q = np.zeros((GRID_W*GRID_H, 4), dtype=np.float32)
alpha, gamma = 0.3, 0.9
epsilon, min_eps, decay = 0.9, 0.05, 0.995

training = False
manual_mode = False
show_q = True
show_table = True
show_coords = True
steps_per_tick = 20.0
step_accum = 0.0
episode = 0
steps = 0
cur = start
ep_reward = 0

# histories for graphs
ep_rewards_history = []
epsilon_history = []
steps_per_episode_history = []
max_q_history = []

table_page = 0

# clickable UI actions populated each frame
click_actions = []  # list of tuples (pygame.Rect, callable)

# helpers to adjust parameters via mouse

def adjust_alpha(delta):
    global alpha
    alpha = max(0.05, min(1.0, alpha + delta))

def adjust_gamma(delta):
    global gamma
    gamma = max(0.0, min(0.99, gamma + delta))

def adjust_epsilon(delta):
    global epsilon
    epsilon = max(min_eps, min(1.0, epsilon + delta))

def toggle_decay():
    global decay
    decay = 1.0 if abs(decay-1.0) > 1e-6 else 0.995

def toggle_training():
    global training
    training = not training
    # turning training on disables manual mode
    if training:
        set_manual_mode(False)

def toggle_show_q():
    global show_q
    show_q = not show_q

def toggle_show_table():
    global show_table
    show_table = not show_table

def toggle_show_coords():
    global show_coords
    show_coords = not show_coords

def set_manual_mode(on: bool):
    global manual_mode, training
    manual_mode = on
    if on:
        training = False

def adjust_speed(delta):
    global steps_per_tick
    steps_per_tick = max(0.5, min(200.0, float(steps_per_tick) + float(delta)))

def page_prev():
    global table_page
    table_page = max(0, table_page - 1)

def page_next():
    global table_page
    table_page = table_page + 1

# ---- Helpers ----
def idx(x,y):
    return y*GRID_W + x

def in_bounds(x,y):
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def step(state, action_idx):
    x,y = state
    dx,dy = ACTIONS[action_idx]
    nx,ny = x+dx, y+dy
    # out of bounds or into a wall = blocked, cost a step
    if not in_bounds(nx,ny) or (nx,ny) in walls:
        return (x,y), STEP_REWARD, False
    s = (nx,ny)
    # obstacle: lose points and return to start, episode continues
    if s in holes:
        return start, HOLE_REWARD, False
    if s == goal:
        return s, GOAL_REWARD, True
    return s, STEP_REWARD, False

# epsilon-greedy
def choose_action(s):
    if random.random() < epsilon:
        return random.randrange(4)
    si = idx(*s)
    return int(np.argmax(q[si]))

def reset_episode():
    global cur, ep_reward
    cur = start
    ep_reward = 0

# ---- Rendering ----
def draw_grid():
    screen.fill(BLACK)
    for y in range(GRID_H):
        for x in range(GRID_W):
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            color = GRID
            if (x,y) in walls: color = WALL
            if (x,y) in holes: color = HOLE
            if (x,y) == goal: color = GOAL
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GREY, rect, 1)
            if show_coords:
                txt = small.render(f"{x},{y}", True, (180,180,180))
                screen.blit(txt, (x*CELL+6, y*CELL+4))

    # start marker
    sx,sy = start
    pygame.draw.rect(screen, (70,70,140), pygame.Rect(sx*CELL+10, sy*CELL+10, CELL-20, CELL-20), 2)

    # agent
    ax,ay = cur
    pygame.draw.circle(screen, AGENT, (ax*CELL+CELL//2, ay*CELL+CELL//2), CELL//3)

    # right sidebar (HUD + Q-table)
    pygame.draw.rect(screen, (25,25,25), pygame.Rect(GRID_W*CELL, 0, SIDEBAR, GRID_H*CELL))
    # bottom panel (graphs)
    pygame.draw.rect(screen, (22,22,22), pygame.Rect(0, GRID_H*CELL, W, BOTTOM))


def arrow_for(max_a, cx, cy, strength):
    # draw arrow aligned to current ACTIONS mapping
    dx, dy = ACTIONS[max_a]
    col = (int(40+200*strength), int(40+200*(1-strength)), 60)
    if dy == -1:  # up
        pygame.draw.polygon(screen, col, [(cx,cy-18),(cx-8,cy-2),(cx+8,cy-2)])
    elif dx == 1:  # right
        pygame.draw.polygon(screen, col, [(cx+18,cy),(cx+2,cy-8),(cx+2,cy+8)])
    elif dy == 1:  # down
        pygame.draw.polygon(screen, col, [(cx,cy+18),(cx-8,cy+2),(cx+8,cy+2)])
    elif dx == -1:  # left
        pygame.draw.polygon(screen, col, [(cx-18,cy),(cx-2,cy-8),(cx-2,cy+8)])


def draw_q_overlay():
    for y in range(GRID_H):
        for x in range(GRID_W):
            if (x,y) in walls: continue
            si = idx(x,y)
            arr = q[si]
            mmin, mmax = float(arr.min()), float(arr.max())
            a = int(np.argmax(arr))
            strength = 0.0 if mmax<=0 else min(1.0, mmax/(GOAL_REWARD))
            cx, cy = x*CELL+CELL//2, y*CELL+CELL//2
            arrow_for(a, cx, cy, strength)
            if (x,y) == cur:
                # show numeric q for current state
                txt = small.render(f"Q: {arr.round(2)}", True, TEXT)
                screen.blit(txt, (x*CELL+6, y*CELL+6))


# ... helper to draw a simple button and return its rect
def draw_button(x, y, w, h, label):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, (45,45,45), rect)
    pygame.draw.rect(screen, GREY, rect, 1)
    text = small.render(label, True, TEXT)
    screen.blit(text, (x + (w - text.get_width())//2, y + (h - text.get_height())//2))
    return rect

def copy_q_snapshot():
    try:
        with open("q_snapshot.txt", "w", encoding="utf-8") as f:
            for y in range(GRID_H):
                for x in range(GRID_W):
                    if (x,y) in walls: continue
                    arr = q[idx(x,y)]
                    f.write(f"({x},{y}): [{arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f}, {arr[3]:.3f}]\n")
        print("Saved Q snapshot to q_snapshot.txt")
    except Exception as e:
        print(f"Failed to save snapshot: {e}")


def clear_q():
    global q
    q[:] = 0

def hud():
    x0 = GRID_W*CELL + 12
    y0 = 16
    lines = [
        "Q-Learning Demo",
        f"episodes: {episode}",
        f"steps: {steps}",
        f"alpha: {alpha:.2f}",
        f"gamma: {gamma:.2f}",
        f"epsilon: {epsilon:.2f} (min {min_eps:.2f})",
        f"decay: {decay:.3f}",
        f"ep_reward: {ep_reward:.1f}",
        f"avg last 20: {np.mean(ep_rewards_history[-20:]) if ep_rewards_history[-20:] else 0:.1f}",
        f"maxQ: {float(np.max(q)):.2f}",
        f"mode: {'manual' if manual_mode else ('training' if training else 'paused')}",
        f"speed: {steps_per_tick:.2f} steps/tick",
    ]
    for i,l in enumerate(lines):
        screen.blit(font.render(l, True, TEXT), (x0, y0 + i*24))

    y1 = y0 + len(lines)*24 + 8
    eq = font.render("Update: Q[s,a]+=alpha*(r+gamma*maxQ[s']-Q[s,a])", True, TEXT)
    screen.blit(eq, (x0, y1))

    btn_y = y1 + 30
    r_train = draw_button(x0, btn_y, 120, 28, "Pause" if training else "Play")
    click_actions.append((r_train, toggle_training))

    r_step = draw_button(x0+130, btn_y, 140, 28, "Step (ε-greedy)")
    click_actions.append((r_step, lambda: (set_manual_mode(True), perform_step(None))))

    # Tick control buttons (speed adjustment)
    tick_y = btn_y + 35
    r_slower = draw_button(x0, tick_y, 55, 28, "<<")
    click_actions.append((r_slower, lambda: adjust_speed(-5)))
    
    r_faster = draw_button(x0+65, tick_y, 55, 28, ">>")
    click_actions.append((r_faster, lambda: adjust_speed(5)))
    
    r_reset_speed = draw_button(x0+130, tick_y, 70, 28, "Reset")
    click_actions.append((r_reset_speed, lambda: globals().update(steps_per_tick=20.0)))

    r_half = draw_button(x0+210, tick_y, 55, 28, "0.5")
    click_actions.append((r_half, lambda: globals().update(steps_per_tick=0.5)))

    r_copy = draw_button(x0+270, tick_y, 70, 28, "Copy")
    click_actions.append((r_copy, copy_q_snapshot))

    r_delete = draw_button(x0+350, tick_y, 70, 28, "Delete")
    click_actions.append((r_delete, clear_q))

    # Q-table toggle button
    toggle_y = tick_y + 35
    r_toggle_qtable = draw_button(x0, toggle_y, 120, 28, "Hide Q-table" if show_table else "Show Q-table")
    click_actions.append((r_toggle_qtable, toggle_show_table))

    # Complete Q-table (all states, no pagination) - only show if enabled
    if show_table:
        mini_x = x0
        mini_y = toggle_y + 38
        mini_w = SIDEBAR - 24
        
        # Calculate height needed for all states
        states = [(sx,sy) for sy in range(GRID_H) for sx in range(GRID_W) if (sx,sy) not in walls]
        needed_height = 28 + len(states) * 18 + 10  # header + states + padding
        
        pygame.draw.rect(screen, (28,28,28), pygame.Rect(mini_x, mini_y, mini_w, needed_height))
        pygame.draw.rect(screen, GREY, pygame.Rect(mini_x, mini_y, mini_w, needed_height), 1)

        header = small.render(f"Complete Q-table ({len(states)} states)", True, TEXT)
        screen.blit(header, (mini_x + 6, mini_y + 6))

        y = mini_y + 28
        for (sx,sy) in states:
            arr = q[idx(sx,sy)]
            # Show all 4 Q-values: [left, right, up, down]
            q_str = f"[{arr[0]:.1f},{arr[1]:.1f},{arr[2]:.1f},{arr[3]:.1f}]"
            line = small.render(f"({sx},{sy}): {q_str}", True, TEXT)
            screen.blit(line, (mini_x + 6, y))
            y += 18

# ---- Graphs and Q-table rendering ----
def draw_series(rect, data, color=(120,180,240), max_points=120, min_y=None, max_y=None):
    x, y, w, h = rect
    pygame.draw.rect(screen, (30,30,30), rect)
    pygame.draw.rect(screen, GREY, rect, 1)
    if not data:
        return
    ds = data[-max_points:]
    n = len(ds)
    my = min(ds) if min_y is None else min_y
    My = max(ds) if max_y is None else max_y
    if abs(My-my) < 1e-6:
        My = my + 1.0
    pts = []
    for i,val in enumerate(ds):
        px = x + int(i*(w-6)/max(1,n-1)) + 3
        py = y + h - 3 - int((val-my)/(My-my) * (h-6))
        pts.append((px,py))
    if len(pts) >= 2:
        pygame.draw.lines(screen, color, False, pts, 2)


def draw_graphs():
    # graphs occupy the top portion of the bottom panel
    x0 = 10
    y0 = GRID_H*CELL + 10
    total_w = W - 20
    total_h = BOTTOM - 20
    graphs_h = int(total_h * 0.6)
    col_w = (total_w - 20) // 3
    g1 = (x0, y0, col_w, graphs_h)
    g2 = (x0 + col_w + 10, y0, col_w, graphs_h)
    g3 = (x0 + 2*(col_w + 10), y0, col_w, graphs_h)

    draw_series(g1, ep_rewards_history, (200,140,60))
    screen.blit(small.render("episode rewards", True, TEXT), (g1[0]+6, g1[1]-18))

    draw_series(g2, epsilon_history, (140,200,60), min_y=0.0, max_y=1.0)
    screen.blit(small.render("epsilon", True, TEXT), (g2[0]+6, g2[1]-18))

    draw_series(g3, max_q_history, (120,180,240))
    screen.blit(small.render("max Q", True, TEXT), (g3[0]+6, g3[1]-18))


def draw_q_table():
    # render Q-table below graphs within the bottom panel
    base_x = 10
    base_y = GRID_H*CELL + 10
    total_w = W - 20
    total_h = BOTTOM - 20
    graphs_h = int(total_h * 0.6)

    x0 = base_x
    y0 = base_y + graphs_h + 10
    w = total_w
    h = total_h - graphs_h - 10

    panel = pygame.Rect(x0, y0, w, h)
    pygame.draw.rect(screen, (28,28,28), panel)
    pygame.draw.rect(screen, GREY, panel, 1)

    states = [(x,y) for y in range(GRID_H) for x in range(GRID_W) if (x,y) not in walls]
    rows = max(6, h//20 - 1)
    total_pages = max(1, (len(states) + rows - 1)//rows)
    page = max(0, min(table_page, total_pages-1))
    start = page*rows
    shown = states[start:start+rows]

    header = font.render(f"Q-table {page+1}/{total_pages} (Up/Down)", True, TEXT)
    screen.blit(header, (x0+6, y0+8))

    # mouse pagination removed; keyboard Up/Down controls remain.

    y = y0 + 36
    for (sx,sy) in shown:
        arr = q[idx(sx,sy)]
        line = small.render(f"({sx},{sy}) -> {arr.round(2)}", True, TEXT)
        screen.blit(line, (x0+6, y))
        y += 20

# ---- Training step ----
def train_tick():
    global cur, epsilon, episode, steps, ep_reward, step_accum
    step_accum += steps_per_tick
    nsteps = int(step_accum)
    if nsteps <= 0:
        return
    step_accum -= nsteps
    for _ in range(nsteps):
        a = choose_action(cur)
        nxt, r, done = step(cur, a)
        si = idx(*cur)
        ni = idx(*nxt)
        td_target = r + gamma * (0 if done else float(np.max(q[ni])))
        q[si,a] += alpha * (td_target - q[si,a])
        cur = nxt
        ep_reward += r
        steps += 1
        if done:
            episode += 1
            if epsilon > min_eps:
                epsilon = max(min_eps, epsilon*decay)
            ep_rewards_history.append(ep_reward)
            epsilon_history.append(epsilon)
            steps_per_episode_history.append(steps)
            max_q_history.append(float(np.max(q)))
            reset_episode()
            break

# ---- One episode (blocking short) ----
def run_one_episode(max_steps=500):
    global cur, epsilon, episode, steps, ep_reward
    reset_episode()
    for _ in range(max_steps):
        a = choose_action(cur)
        nxt, r, done = step(cur, a)
        si = idx(*cur); ni = idx(*nxt)
        td_target = r + gamma * (0 if done else float(np.max(q[ni])))
        q[si,a] += alpha * (td_target - q[si,a])
        cur = nxt; ep_reward += r; steps += 1
        if done:
            episode += 1
            if epsilon > min_eps:
                epsilon = max(min_eps, epsilon*decay)
            ep_rewards_history.append(ep_reward)
            epsilon_history.append(epsilon)
            steps_per_episode_history.append(steps)
            max_q_history.append(float(np.max(q)))
            break

# ---- Manual single step ----
def perform_step(action_idx=None):
    """Apply one environment step and Q update. If action_idx is None, use epsilon-greedy."""
    global cur, epsilon, episode, steps, ep_reward
    a = choose_action(cur) if action_idx is None else action_idx
    nxt, r, done = step(cur, a)
    si = idx(*cur); ni = idx(*nxt)
    td_target = r + gamma * (0 if done else float(np.max(q[ni])))
    q[si,a] += alpha * (td_target - q[si,a])
    cur = nxt; ep_reward += r; steps += 1
    if done:
        episode += 1
        if epsilon > min_eps:
            epsilon = max(min_eps, epsilon*decay)
        ep_rewards_history.append(ep_reward)
        epsilon_history.append(epsilon)
        steps_per_episode_history.append(steps)
        max_q_history.append(float(np.max(q)))
        reset_episode()

# ---- Main loop ----
reset_episode()
while True:
    events = pygame.event.get()
    # Re-enable mouse handling for minimal buttons
    mouse_clicks = [e for e in events if e.type == pygame.MOUSEBUTTONDOWN]
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit(0)
        # handle window resize
        if event.type == pygame.VIDEORESIZE:
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            compute_layout()
        if event.type == pygame.KEYDOWN:
             if event.key == pygame.K_SPACE:
                 training = not training
             elif event.key == pygame.K_q:
                 show_q = not show_q
             elif event.key == pygame.K_r:
                 q[:] = 0; episode=0; steps=0; epsilon=0.9; reset_episode()
             elif event.key == pygame.K_c:
                 q[:] = 0
             elif event.key == pygame.K_d:
                 decay = 1.0 if abs(decay-1.0)>1e-6 else 0.995
             elif event.key == pygame.K_1:
                 alpha = max(0.05, alpha-0.05)
             elif event.key == pygame.K_2:
                 alpha = min(1.0, alpha+0.05)
             elif event.key == pygame.K_3:
                 gamma = max(0.0, gamma-0.05)
             elif event.key == pygame.K_4:
                 gamma = min(0.99, gamma+0.05)
             elif event.key == pygame.K_5:
                 epsilon = max(min_eps, epsilon-0.05)
             elif event.key == pygame.K_6:
                 epsilon = min(1.0, epsilon+0.05)
             elif event.key == pygame.K_n:
                 run_one_episode()
             elif event.key == pygame.K_t:
                 show_table = not show_table
             elif event.key == pygame.K_m:
                 set_manual_mode(not manual_mode)
             elif event.key == pygame.K_k:
                 show_coords = not show_coords
             elif event.key == pygame.K_UP:
                 table_page = max(0, table_page-1)
             elif event.key == pygame.K_DOWN:
                 table_page = table_page+1
             # manual action mapping per spec
             elif manual_mode and event.key in (pygame.K_w, pygame.K_UP):
                 perform_step(2)
             elif manual_mode and event.key in (pygame.K_d, pygame.K_RIGHT):
                 perform_step(1)
             elif manual_mode and event.key in (pygame.K_s, pygame.K_DOWN):
                 perform_step(3)
             elif manual_mode and event.key in (pygame.K_a, pygame.K_LEFT):
                 perform_step(0)
             elif manual_mode and event.key == pygame.K_s:
                 perform_step(None)

    if training:
        train_tick()

    # reset clickable registry for this frame
    click_actions = []

    draw_grid()
    if show_q:
        draw_q_overlay()
    if show_table:
        draw_q_table()
    draw_graphs()
    hud()

    # process mouse clicks against registered buttons
    for e in mouse_clicks:
        for rect, fn in click_actions:
            if rect.collidepoint(e.pos):
                try:
                    fn()
                except Exception:
                    pass
                break

    pygame.display.flip()
    clock.tick(60)