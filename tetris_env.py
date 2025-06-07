import pygame
import numpy as np
import cv2
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import os
from tetris import *

class TetrisImageEnv:
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode
        
        # Wymiary dla przetwarzania obrazu
        self.screen_width = 400
        self.screen_height = 600
        self.processed_width = 84
        self.processed_height = 84
        
        # Inicjalizuj pygame i screen przed reset()
        pygame.init()
        pygame.font.init()
        if render_mode == 'human':
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Tetris RL')
        else:
            # Tryb bez wyświetlania - tylko przechwytywanie
            self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        self.font = pygame.font.SysFont('arial', 20)
        self.clock = pygame.time.Clock()
        
        # Teraz można bezpiecznie wywołać reset()
        self.reset()

    def __del__(self):
        print(">>> DEL: Destroying TetrisEnv, quitting Pygame")
        #pygame.quit()
        
    def reset(self):
        """Reset środowiska do stanu początkowego"""
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.score = 0
        self.lines_cleared = 0
        self.done = False
        self.fall_time = 0
        self.fall_speed = 500  # milliseconds
        self.game_time = 0
        
        return self.get_screen_observation()
    
    def get_screen_observation(self):
        """Pobiera screenshot i przetwarza go do formatu dla CNN"""
        # Renderuj aktualny stan gry
        self.render()
        
        # Pobierz surowy obraz
        raw_screen = pygame.surfarray.array3d(self.screen)
        raw_screen = raw_screen.transpose((1, 0, 2))  # pygame ma odwróconą orientację
        
        # Konwersja do grayscale
        gray_screen = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2GRAY)
        
        # Resize do standardowego rozmiaru
        processed_screen = cv2.resize(gray_screen, (self.processed_width, self.processed_height))
        
        # Normalizacja pikseli do zakresu [0, 1]
        processed_screen = processed_screen.astype(np.float32) / 255.0
        
        return processed_screen
    
    def step(self, action):
        """
        Wykonaj akcję w środowisku
        Actions: 0=left, 1=right, 2=rotate, 3=soft_drop, 4=hard_drop, 5=do_nothing
        """
        if self.done:
            return self.get_screen_observation(), 0, self.done, {}
        
        reward = 0
        old_score = self.score
        self.game_time += 1
        
        # Wykonaj akcję
        if action == 0:  # Left
            self.current_piece.x -= 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x += 1
                
        elif action == 1:  # Right
            self.current_piece.x += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x -= 1
                
        elif action == 2:  # Rotate
            old_rotation = self.current_piece.rotation
            self.current_piece.rotation = (self.current_piece.rotation + 1) % len(self.current_piece.shape)
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.rotation = old_rotation
                
        elif action == 3:  # Soft drop
            self.current_piece.y += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
            else:
                reward += 1  # Mała nagroda za soft drop
                
        elif action == 4:  # Hard drop
            drop_distance = 0
            while valid_space(self.current_piece, self.grid):
                self.current_piece.y += 1
                drop_distance += 1
            self.current_piece.y -= 1
            reward += drop_distance * 2  # Nagroda za hard drop
            
        # Action 5 (do nothing) nie wymaga kodu
        
        # Aktualizuj grid
        self.grid = create_grid(self.locked_positions)
        shape_pos = convert_shape_format(self.current_piece)
        
        # Umieść aktualny klocek na gridzie
        for x, y in shape_pos:
            if y > -1:
                self.grid[y][x] = self.current_piece.color
        
        # Sprawdź czy klocek powinien zostać zablokowany
        if self._should_lock_piece():
            # Zablokuj klocek
            for pos in shape_pos:
                if pos[1] > -1:
                    self.locked_positions[(pos[0], pos[1])] = self.current_piece.color
            
            # Nowy klocek
            self.current_piece = self.next_piece
            self.next_piece = get_shape()
            
            # Sprawdź usunięte linie
            lines_cleared = clear_rows(self.grid, self.locked_positions)
            self.lines_cleared += lines_cleared
            
            # Ulepszone nagrody
            if lines_cleared > 0:
                # Eksponencjalna nagroda za więcej linii jednocześnie
                line_rewards = {1: 40, 2: 100, 3: 300, 4: 1200}
                reward += line_rewards.get(lines_cleared, 0)
                self.score += lines_cleared * 10
            
            # Nagroda za wysokość umieszczenia (preferuj niższe umieszczenie)
            avg_height = sum(pos[1] for pos in shape_pos) / len(shape_pos)
            height_reward = max(0, 20 - avg_height)  # Im niżej, tym lepiej
            reward += height_reward
            
            # Sprawdź czy gra się skończyła
            if check_lost(self.locked_positions):
                self.done = True
                reward -= 100  # Kara za przegraną
        
        # Mała nagroda za przetrwanie
        if not self.done:
            reward += 1
        
        info = {
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'level': self.lines_cleared // 10 + 1,
            'game_time': self.game_time
        }
        
        return self.get_screen_observation(), reward, self.done, info
    
    def _should_lock_piece(self):
        """Sprawdź czy klocek powinien zostać zablokowany"""
        self.current_piece.y += 1
        can_move = valid_space(self.current_piece, self.grid)
        self.current_piece.y -= 1
        return not can_move
    
    def render(self):
        """Renderuj aktualny stan gry"""
        self.screen.fill((0, 0, 0))  # Czarne tło
        
        # Rysuj grid
        start_x = 50
        start_y = 50
        cell_size = 25
        
        # Rysuj zajęte komórki
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                color = self.grid[i][j]
                if color != (0, 0, 0):
                    pygame.draw.rect(self.screen, color,
                                   (start_x + j * cell_size, 
                                    start_y + i * cell_size,
                                    cell_size, cell_size))
        
        # Rysuj linie siatki
        for i in range(21):  # 21 linii poziomych
            pygame.draw.line(self.screen, (64, 64, 64),
                           (start_x, start_y + i * cell_size),
                           (start_x + 10 * cell_size, start_y + i * cell_size))
        
        for j in range(11):  # 11 linii pionowych
            pygame.draw.line(self.screen, (64, 64, 64),
                           (start_x + j * cell_size, start_y),
                           (start_x + j * cell_size, start_y + 20 * cell_size))
        
        # Rysuj informacje
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        lines_text = self.font.render(f'Lines: {self.lines_cleared}', True, (255, 255, 255))
        
        self.screen.blit(score_text, (300, 50))
        self.screen.blit(lines_text, (300, 80))
        
        # Rysuj następny klocek
        next_text = self.font.render('Next:', True, (255, 255, 255))
        self.screen.blit(next_text, (300, 120))
        
        if self.next_piece:
            shape_format = self.next_piece.shape[0]  # Pierwsza rotacja
            for i, line in enumerate(shape_format):
                for j, char in enumerate(line):
                    if char == '0':
                        pygame.draw.rect(self.screen, self.next_piece.color,
                                       (300 + j * 15, 150 + i * 15, 15, 15))
        
        if self.render_mode == 'human':
            pygame.display.flip()
    
    def close(self):
        """Zamknij środowisko"""
        #pygame.quit()


class StackedFramesDQN(nn.Module):
    """DQN z możliwością przetwarzania kilku klatek jednocześnie"""
    def __init__(self, input_shape, n_actions, n_frames=4):
        super(StackedFramesDQN, self).__init__()
        
        self.n_frames = n_frames
        
        # CNN layers - dostosowane do większej liczby kanałów
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Oblicz rozmiar po convolution
        conv_out_size = self._get_conv_out((n_frames, input_shape[1], input_shape[2]))
        
        # Dueling DQN architecture
        self.advantage = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_out(self, shape):
        """Oblicz rozmiar wyjścia z warstw konwolucyjnych"""
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        
        advantage = self.advantage(x)
        value = self.value(x)
        
        # Dueling DQN formula
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities.append(max_priority)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(list(self.priorities)[:len(self.buffer)])
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)


class ImprovedTetrisAgent:
    def __init__(self, state_shape, n_actions, lr=0.0001, n_frames=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.n_actions = n_actions
        self.n_frames = n_frames
        
        # Sieci neuronowe z Dueling DQN
        self.q_net = StackedFramesDQN(state_shape, n_actions, n_frames).to(self.device)
        self.target_net = StackedFramesDQN(state_shape, n_actions, n_frames).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer z learning rate scheduling
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(50000)
        
        # Frame stacking
        self.frame_stack = deque(maxlen=n_frames)
        
        # Hyperparameters - dostrojone
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Wolniejszy decay
        self.gamma = 0.99
        self.batch_size = 32
        self.target_update = 500  # Częstsze aktualizacje
        self.steps = 0
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'scores': [],
            'rewards': [],
            'epsilon': [],
            'losses': [],
            'learning_rates': []
        }
        
    def preprocess_state(self, state):
        """Przetwórz stan i dodaj do stack"""
        if len(self.frame_stack) == 0:
            # Wypełnij stack pierwszą klatką
            for _ in range(self.n_frames):
                self.frame_stack.append(state)
        else:
            self.frame_stack.append(state)
            
        return np.stack(self.frame_stack, axis=0)
        
    def reset_frame_stack(self):
        """Reset frame stack na początku epizodu"""
        self.frame_stack.clear()
        
    def remember(self, state, action, reward, next_state, done):
        """Zapisz doświadczenie w prioritized replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
        
    def act(self, state, training=True):
        """Wybierz akcję używając epsilon-greedy z noise injection"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.n_actions)
        
        # Konwertuj state do tensora
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_net(state_tensor)
        
        if training:
            # Dodaj szum do eksploracji
            noise = torch.randn_like(q_values) * 0.1 * self.epsilon
            q_values += noise
            
        return q_values.argmax().item()
        
    def replay(self):
        """Trenuj sieć na batch z prioritized replay buffer"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample z prioritized replay
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Konwertuj do tensorów
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_actions = self.q_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # TD errors dla prioritized replay
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values).detach().cpu().numpy()
        
        # Weighted loss
        loss = (weights * F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10)  # Gradient clipping
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors + 1e-6)
        
        # Aktualizuj epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Aktualizuj target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        return loss.item()
        
    def save(self, filename):
        """Zapisz model i statystyki"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_stats': self.training_stats
        }, filename)
        
    def load(self, filename):
        """Wczytaj model i statystyki"""
        checkpoint = torch.load(filename)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.training_stats = checkpoint.get('training_stats', self.training_stats)


class TrainingMonitor:
    """Monitor postępów treningu"""
    def __init__(self, save_dir='./training_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.episode_scores = []
        self.episode_rewards = []
        self.losses = []
        self.epsilons = []
        
    def log_episode(self, episode, score, total_reward, epsilon, loss=None):
        """Zaloguj statystyki epizodu"""
        self.episode_scores.append(score)
        self.episode_rewards.append(total_reward)
        self.epsilons.append(epsilon)
        if loss is not None:
            self.losses.append(loss)
            
        # Co 100 epizodów zapisz wykresy
        if episode % 100 == 0:
            self.plot_progress(episode)
            self.save_stats(episode)
            
    def plot_progress(self, episode):
        """Rysuj wykresy postępów"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scores
        axes[0, 0].plot(self.episode_scores)
        axes[0, 0].set_title('Episode Scores')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Score')
        
        # Running average scores
        if len(self.episode_scores) >= 100:
            running_avg = [np.mean(self.episode_scores[max(0, i-99):i+1]) 
                          for i in range(len(self.episode_scores))]
            axes[0, 1].plot(running_avg)
            axes[0, 1].set_title('Running Average Score (100 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Score')
        
        # Rewards
        axes[1, 0].plot(self.episode_rewards)
        axes[1, 0].set_title('Episode Total Rewards')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Total Reward')
        
        # Epsilon
        axes[1, 1].plot(self.epsilons)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/progress_episode_{episode}.png')
        plt.close()
        
    def save_stats(self, episode):
        """Zapisz statystyki do pliku JSON"""
        stats = {
            'episode': episode,
            'scores': self.episode_scores,
            'rewards': self.episode_rewards,
            'epsilons': self.epsilons,
            'losses': self.losses,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'{self.save_dir}/training_stats_{episode}.json', 'w') as f:
            json.dump(stats, f)


def train_improved_tetris_agent(episodes=5000, render_every=500):
    """Ulepszona pętla treningowa z monitoringiem"""
    
    # Stwórz folder na wyniki
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./tetris_training_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    env = TetrisImageEnv(render_mode='headless')
    monitor = TrainingMonitor(save_dir)
    
    state_shape = (1, 84, 84)
    n_actions = 6
    n_frames = 4
    
    agent = ImprovedTetrisAgent(state_shape, n_actions, n_frames=n_frames)
    
    best_score = 0
    best_avg_score = 0
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Results will be saved to: {save_dir}")
    
    start_time = time.time()
    
    for episode in range(episodes):
        agent.reset_frame_stack()
        raw_state = env.reset()
        state = agent.preprocess_state(raw_state)
        
        total_reward = 0
        episode_loss = []
        
        while True:
            action = agent.act(state, training=True)
            next_raw_state, reward, done, info = env.step(action)
            next_state = agent.preprocess_state(next_raw_state)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Trenuj agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_loss.append(loss)
                    
            if done:
                break
        
        # Statystyki epizodu
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        monitor.log_episode(episode, info['score'], total_reward, agent.epsilon, avg_loss)
        
        # Zapisz najlepszy model
        if info['score'] > best_score:
            best_score = info['score']
            agent.save(f'{save_dir}/best_model_score_{best_score}.pth')
        
        # Sprawdź średni wynik z ostatnich 100 epizodów
        if episode >= 100:
            recent_scores = monitor.episode_scores[-100:]
            avg_score = np.mean(recent_scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                agent.save(f'{save_dir}/best_avg_model_{avg_score:.1f}.pth')
        
        # Progress report
        if episode % 50 == 0:
            elapsed = time.time() - start_time
            avg_score_100 = np.mean(monitor.episode_scores[-100:]) if len(monitor.episode_scores) >= 100 else 0
            
            print(f"Episode {episode:4d} | "
                  f"Score: {info['score']:3d} | "
                  f"Reward: {total_reward:6.1f} | "
                  f"Lines: {info['lines_cleared']:2d} | "
                  f"Avg100: {avg_score_100:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed/60:.1f}min")
        
        # Renderuj co kilka epizodów
        if episode % render_every == 0 and episode > 0:
            print(f"\n--- Rendering episode {episode} ---")
            test_agent_visual(agent, episodes=1)
            
        # Checkpoint co 1000 epizodów
        if episode % 1000 == 0 and episode > 0:
            agent.save(f'{save_dir}/checkpoint_episode_{episode}.pth')
    
    # Finalne zapisanie
    agent.save(f'{save_dir}/final_model.pth')
    
    print(f"\nTraining completed!")
    print(f"Best score: {best_score}")
    print(f"Best average score (100 ep): {best_avg_score:.1f}")
    print(f"Results saved to: {save_dir}")
    
    env.close()
    return agent, save_dir


def test_agent_visual(agent, episodes=5):
    """Test agenta z wizualizacją"""
    env = TetrisImageEnv(render_mode='human')
    
    test_scores = []
    
    for episode in range(episodes):
        agent.reset_frame_stack()
        raw_state = env.reset()
        state = agent.preprocess_state(raw_state)
        
        total_score = 0
        step_count = 0
        
        while True:
            action = agent.act(state, training=False)  # Bez eksploracji
            next_raw_state, reward, done, info = env.step(action)
            next_state = agent.preprocess_state(next_raw_state)
            
            state = next_state
            total_score += reward
            step_count += 1
            
            env.clock.tick(10)  # Ograniczenie FPS
            
            if done:
                test_scores.append(info['score'])
                print(f"Test Episode {episode + 1}: Score = {info['score']}, Steps = {step_count}")
                break
        
        if episode < episodes - 1:  # Pauza między grami (oprócz ostatniej)
            pygame.time.wait(2000)
    
    env.close()
    
    if test_scores:
        print(f"\nTest Results:")
        print(f"Average Score: {np.mean(test_scores):.1f}")
        print(f"Best Score: {max(test_scores)}")
        print(f"Scores: {test_scores}")
    
    return test_scores


# Główny kod wykonawczy
if __name__ == "__main__":
    print("=== Tetris RL Training System ===")
    print("1. Train new agent")
    print("2. Continue training from checkpoint")
    print("3. Test trained agent")
    print("4. Load and test specific model")
    
    choice = input("Choose option (1-4): ").strip()
    
    if choice == "1":
        # Nowy trening
        print("\nStarting new training...")
        episodes = int(input("Enter number of episodes (default 5000): ") or "5000")
        render_freq = int(input("Render every N episodes (default 500): ") or "500")
        
        agent, save_dir = train_improved_tetris_agent(episodes=episodes, render_every=render_freq)
        
        print(f"\nTraining completed! Results saved to: {save_dir}")
        
        # Test po treningu
        test_choice = input("Test the trained agent now? (y/n): ").strip().lower()
        if test_choice == 'y':
            print("\nTesting trained agent...")
            test_agent_visual(agent, episodes=3)
    
    elif choice == "2":
        # Kontynuacja treningu
        model_path = input("Enter path to checkpoint file: ").strip()
        
        try:
            # Stwórz agenta i załaduj checkpoint
            env = TetrisImageEnv(render_mode='headless')
            state_shape = (1, 84, 84)
            n_actions = 6
            n_frames = 4
            
            agent = ImprovedTetrisAgent(state_shape, n_actions, n_frames=n_frames)
            agent.load(model_path)
            
            print(f"Loaded model from: {model_path}")
            print(f"Current epsilon: {agent.epsilon}")
            print(f"Training steps: {agent.steps}")
            
            additional_episodes = int(input("Additional episodes to train: ") or "1000")
            
            # Kontynuuj trening z załadowanego modelu
            # Tutaj trzeba by zmodyfikować funkcję train_improved_tetris_agent
            # żeby przyjmowała istniejącego agenta
            print("Continuing training...")
            
            env.close()
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    elif choice == "3":
        # Test najnowszego modelu
        import glob
        import os
        
        # Znajdź najnowszy folder z treningu
        training_folders = glob.glob('./tetris_training_*')
        if not training_folders:
            print("No training folders found!")
            exit()
        
        latest_folder = max(training_folders, key=os.path.getctime)
        
        # Znajdź najlepszy model
        model_files = glob.glob(f'{latest_folder}/best_*.pth')
        if not model_files:
            model_files = glob.glob(f'{latest_folder}/final_model.pth')
        
        if not model_files:
            print("No model files found!")
            exit()
        
        model_path = model_files[0]
        print(f"Loading model: {model_path}")
        
        # Załaduj i testuj
        state_shape = (1, 84, 84)
        n_actions = 6
        n_frames = 4
        
        agent = ImprovedTetrisAgent(state_shape, n_actions, n_frames=n_frames)
        agent.load(model_path)
        agent.epsilon = 0  # Wyłącz eksplorację
        
        test_episodes = int(input("Number of test episodes (default 5): ") or "5")
        print(f"\nTesting agent for {test_episodes} episodes...")
        
        scores = test_agent_visual(agent, episodes=test_episodes)
        
    elif choice == "4":
        # Załaduj konkretny model
        model_path = input("Enter full path to model file: ").strip()
        
        try:
            state_shape = (1, 84, 84)
            n_actions = 6
            n_frames = 4
            
            agent = ImprovedTetrisAgent(state_shape, n_actions, n_frames=n_frames)
            agent.load(model_path)
            agent.epsilon = 0  # Wyłącz eksplorację
            
            print(f"Loaded model from: {model_path}")
            
            test_episodes = int(input("Number of test episodes (default 3): ") or "3")
            scores = test_agent_visual(agent, episodes=test_episodes)
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    else:
        print("Invalid choice!")


def quick_start_training():
    """Szybki start dla nowych użytkowników"""
    print("=== Quick Start Tetris RL Training ===")
    print("This will start training with recommended settings...")
    
    # Automatyczne ustawienia
    episodes = 2000
    render_every = 200
    
    print(f"Episodes: {episodes}")
    print(f"Render every: {render_every} episodes")
    print("Starting in 3 seconds...")
    
    import time
    time.sleep(3)
    
    agent, save_dir = train_improved_tetris_agent(episodes=episodes, render_every=render_every)
    
    print(f"Training completed! Testing the agent...")
    test_agent_visual(agent, episodes=3)
    
    return agent, save_dir


def benchmark_agent(model_path, episodes=100):
    """Benchmark agenta na większej liczbie epizodów"""
    print(f"Benchmarking agent: {model_path}")
    
    state_shape = (1, 84, 84)
    n_actions = 6
    n_frames = 4
    
    agent = ImprovedTetrisAgent(state_shape, n_actions, n_frames=n_frames)
    agent.load(model_path)
    agent.epsilon = 0  # Bez eksploracji
    
    env = TetrisImageEnv(render_mode='headless')  # Szybciej bez renderingu
    
    scores = []
    total_lines = []
    
    print(f"Running {episodes} episodes...")
    
    for episode in range(episodes):
        agent.reset_frame_stack()
        raw_state = env.reset()
        state = agent.preprocess_state(raw_state)
        
        while True:
            action = agent.act(state, training=False)
            next_raw_state, reward, done, info = env.step(action)
            next_state = agent.preprocess_state(next_raw_state)
            state = next_state
            
            if done:
                scores.append(info['score'])
                total_lines.append(info['lines_cleared'])
                break
        
        if (episode + 1) % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print(f"Episode {episode + 1:3d}/{episodes}: Avg Score (last 20): {avg_score:.1f}")
    
    env.close()
    
    # Statystyki
    print(f"\n=== Benchmark Results ===")
    print(f"Episodes: {episodes}")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"Worst Score: {min(scores)}")
    print(f"Score Std: {np.std(scores):.1f}")
    print(f"Average Lines: {np.mean(total_lines):.1f}")
    print(f"Total Lines: {sum(total_lines)}")
    
    return {
        'scores': scores,
        'lines': total_lines,
        'avg_score': np.mean(scores),
        'best_score': max(scores),
        'avg_lines': np.mean(total_lines)
    }


# Dodatkowe funkcje pomocnicze
def create_training_config():
    """Stwórz plik konfiguracyjny dla treningu"""
    config = {
        'training': {
            'episodes': 5000,
            'render_every': 500,
            'save_every': 1000,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'epsilon_decay': 0.9995,
            'gamma': 0.99
        },
        'network': {
            'n_frames': 4,
            'processed_width': 84,
            'processed_height': 84
        },
        'rewards': {
            'line_clear_1': 40,
            'line_clear_2': 100,
            'line_clear_3': 300,
            'line_clear_4': 1200,
            'survival': 1,
            'game_over': -100
        }
    }
    
    with open('tetris_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Configuration saved to tetris_config.json")
    return config
