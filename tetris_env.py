import pygame
import random
from tetris import Piece, get_shape, create_grid, valid_space, convert_shape_format, check_lost, clear_rows

CELL_SIZE = 30
COLS = 10
ROWS = 20
WIDTH = CELL_SIZE * COLS
HEIGHT = CELL_SIZE * ROWS
TOP_LEFT_X = 100
TOP_LEFT_Y = 60

class TetrisEnv:
    def __init__(self):
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.score = 0
        self.fall_speed = 0.5  # spadanie co 0.5 s
        self.fall_time = 0
        self.clock = pygame.time.Clock()
        self.done = False

        pygame.font.init()
        self.font = pygame.font.SysFont('comicsans', 30)

    def reset(self):
        self.locked_positions = {}
        self.grid = create_grid(self.locked_positions)
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.score = 0
        self.done = False
        self.fall_time = 0
        return self.grid

    def step(self, action):
        # action: 0 - left, 1 - right, 2 - rotate, 3 - down, 4 - drop
        if self.done:
            return self.grid, 0, self.done, {}

        reward = 0
        if action == 0:
            self.current_piece.x -= 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x += 1
        elif action == 1:
            self.current_piece.x += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.x -= 1
        elif action == 2:
            self.current_piece.rotation = (self.current_piece.rotation + 1) % len(self.current_piece.shape)
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.rotation = (self.current_piece.rotation - 1) % len(self.current_piece.shape)
        elif action == 3:
            self.current_piece.y += 1
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
        elif action == 4:
            while valid_space(self.current_piece, self.grid):
                self.current_piece.y += 1
            self.current_piece.y -= 1

        self.grid = create_grid(self.locked_positions)
        shape_pos = convert_shape_format(self.current_piece)

        for x, y in shape_pos:
            if y > -1:
                self.grid[y][x] = self.current_piece.color

        if self._should_lock():
            for pos in shape_pos:
                self.locked_positions[(pos[0], pos[1])] = self.current_piece.color
            self.current_piece = self.next_piece
            self.next_piece = get_shape()
            lines_cleared = clear_rows(self.grid, self.locked_positions)
            reward = lines_cleared * 10
            self.grid = create_grid(self.locked_positions)

            if check_lost(self.locked_positions):
                self.done = True
                reward = -10

        return self.grid, reward, self.done, {}

    def _should_lock(self):
        self.current_piece.y += 1
        if not valid_space(self.current_piece, self.grid):
            self.current_piece.y -= 1
            return True
        self.current_piece.y -= 1
        return False

    def draw_grid(self, surface):
        for i in range(ROWS):
            pygame.draw.line(surface, (128, 128, 128), (TOP_LEFT_X, TOP_LEFT_Y + i*CELL_SIZE), (TOP_LEFT_X + WIDTH, TOP_LEFT_Y + i*CELL_SIZE))
        for j in range(COLS):
            pygame.draw.line(surface, (128, 128, 128), (TOP_LEFT_X + j*CELL_SIZE, TOP_LEFT_Y), (TOP_LEFT_X + j*CELL_SIZE, TOP_LEFT_Y + HEIGHT))

    def draw_window(self, surface):
        surface.fill((0, 0, 0))  # czarne tło

        for i in range(ROWS):
            for j in range(COLS):
                color = self.grid[i][j]
                if color != (0, 0, 0):
                    pygame.draw.rect(surface, color, (TOP_LEFT_X + j*CELL_SIZE, TOP_LEFT_Y + i*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        self.draw_grid(surface)

        label = self.font.render('Next:', True, (255, 255, 255))
        surface.blit(label, (TOP_LEFT_X + WIDTH + 20, TOP_LEFT_Y))

        format = self.next_piece.shape[self.next_piece.rotation % len(self.next_piece.shape)]
        for i, line in enumerate(format):
            for j, char in enumerate(line):
                if char == '0':
                    pygame.draw.rect(surface, self.next_piece.color,
                                     (TOP_LEFT_X + WIDTH + 20 + j*CELL_SIZE, TOP_LEFT_Y + 30 + i*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        score_label = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        surface.blit(score_label, (TOP_LEFT_X + WIDTH + 20, TOP_LEFT_Y + 200))

        pygame.draw.rect(surface, (255, 255, 255), (TOP_LEFT_X, TOP_LEFT_Y, WIDTH, HEIGHT), 5)

    def run(self):
        pygame.init()
        win = pygame.display.set_mode((WIDTH + 200, HEIGHT + 100))
        pygame.display.set_caption("Tetris")

        clock = pygame.time.Clock()
        fall_time = 0
        run = True

        while run:
            fall_time += clock.get_rawtime()
            clock.tick()

            if fall_time / 1000 > self.fall_speed:
                fall_time = 0
                self.current_piece.y += 1
                if not valid_space(self.current_piece, self.grid) and self.current_piece.y > 0:
                    self.current_piece.y -= 1
                    shape_pos = convert_shape_format(self.current_piece)
                    for pos in shape_pos:
                        self.locked_positions[(pos[0], pos[1])] = self.current_piece.color
                    self.current_piece = self.next_piece
                    self.next_piece = get_shape()
                    lines_cleared = clear_rows(self.grid, self.locked_positions)
                    self.score += lines_cleared * 10
                    self.grid = create_grid(self.locked_positions)

                    if check_lost(self.locked_positions):
                        run = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.step(0)
                    elif event.key == pygame.K_RIGHT:
                        self.step(1)
                    elif event.key == pygame.K_UP:
                        self.step(2)
                    elif event.key == pygame.K_DOWN:
                        self.step(3)
                    elif event.key == pygame.K_SPACE:
                        self.step(4)

            self.grid = create_grid(self.locked_positions)
            shape_pos = convert_shape_format(self.current_piece)
            for x, y in shape_pos:
                if y > -1:
                    self.grid[y][x] = self.current_piece.color

            self.draw_window(win)
            pygame.display.update()

        pygame.quit()

if __name__ == "__main__":
    env = TetrisEnv()
    env.reset()
    env.run()
