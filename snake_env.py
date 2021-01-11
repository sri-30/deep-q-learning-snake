import pygame
import numpy as np
import random


class SNAKE:
    def __init__(self):
        self.size = self.width, self.height = 300, 300
        x = random.randrange(0, self.width, 10)
        y = random.randrange(0, self.width, 10)
        self.body = np.array(np.array([[x, y], [x + 10, y]]))
        pygame.init()
        self.clock = pygame.time.Clock()
        self.snake_length = 2
        self.direction = [0, -10]
        self.screen = ""
        self.previous_action = 10
        self.reward = 0
        self.food_score = 0
        self.previous_action = -100
        self.action_space = (3, )   # For Non-Traditional Controls (Forward Left Right)
        self.done = False
        self.hunger = 0
        self.food = [random.randrange(0, self.width, 100), random.randrange(0, self.height, 100)]
        if self.food in self.body:
            self.food = np.array([random.randrange(0, self.width, 10), random.randrange(0, self.height, 10)])
        self.food_vector = self.body[0] - self.food
        self.state_shape = self.get_state().shape[1:]

    def dead(self):
        # Called whenever the snake dies
        self.reward -= 300
        self.done = True

    def move(self, move_vector):
        self.body = np.append([self.body[0] + move_vector], self.body, axis=0)
        self.body = self.body[:self.snake_length]
        self.reward = 0

        # Reward given for moving towards and away from the food
        if np.linalg.norm(self.body[1] - self.food) < np.linalg.norm(self.body[0] - self.food):
            self.reward -= 20
        elif np.linalg.norm(self.body[1] - self.food) > np.linalg.norm(self.body[0] - self.food):
            self.reward += 20

        # Reward given for either the x or y co-ordinate equaling the x or y co-ordinate of the food
        if self.food_vector[0] == 0:
            self.reward += 5

        if self.food_vector[1] == 0:
            self.reward += 5

        # Eat food (Check's if head's co-ordinate overlaps with food co-ordinate)
        if self.food[0] == self.body[0][0] and self.food[1] == self.body[0][1]:
            self.eat_food()
            self.gen_food()

    def check_value(self, value):
        # Check if the head of the snake overlaps with any part of the body

        for i in self.body[2:]:
            if value[0] == i[0] and value[1] == i[1]:
                return True
        return False

    def get_state(self):

        self.food_vector = self.body[0] - self.food

        '''state = np.append(self.direction, self.is_blocked())
        state = np.append(state, self.food_vector)
        state = np.reshape(state, [1, 11])'''

        state = np.zeros((10, 10), dtype=int)
        x = 0
        y = 0
        minimum = 100000
        closest_to_food = [0, 0]
        for i in range(10):
            for j in range(10):
                if -50 + i * 10 + self.body[0][0] == self.body[0][0] and -50 + j * 10 + self.body[0][1] == self.body[0][1]:
                    state[i][j] = -2  # Pixel for the head of the snake
                if -50 + i * 10 + self.body[0][1] == -10 or -50 + i * 10 + self.body[0][1] == self.width:
                    state[i][j] = -1   # Vertical Walls
                if -50 + j * 10 + self.body[0][0] == -10 or -50 + j * 10 + self.body[0][0] == self.height:
                    state[i][j] = -1   # Horizontal Walls
                elif self.check_value([-50 + j*10 + self.body[0][0], -50 + i*10 + self.body[0][1]]):
                    state[i][j] = -2   # Snake's Body
                elif -50 + i * 10 + self.body[0][0] == self.body[1][0] and -50 + j * 10 + self.body[0][1] == self.body[1][1]:
                    state[j][i] = -2   # Snake's Body (Previous position of Head)
                if np.linalg.norm(np.array([-50 + i * 10 + self.body[0][0], -50 + j * 10 + self.body[0][1]]) - self.food) < minimum:
                    minimum = np.linalg.norm(np.array([-50 + i * 10 + self.body[0][0], -50 + j * 10 + self.body[0][1]]) - self.food)
                    closest_to_food = [j, i]
        state[closest_to_food[0]][closest_to_food[1]] = 1   # Pixel Closest to food
        state = np.array(state).reshape(-1, 10, 10, 1)
        return state

    def is_blocked(self):
        # Creates an array showing whether there any immediate obstacles in the snake's path in seven directions
        # 2 if there is a wall blocking the snake in a certain direction
        # 1 if the snake's body is blocking the snake in a certain direction
        blocked1 = 0
        blocked2 = 0
        blocked3 = 0
        blocked4 = 0
        blocked5 = 0
        blocked6 = 0
        blocked7 = 0
        front = self.body[0] + self.direction
        left = [0, 0]
        right = [0, 0]

        if self.direction[0] == 10 or self.direction[0] == -10:
            right[0] = self.body[0][0] - self.direction[1]
            right[1] = self.body[0][1] - self.direction[0]
            left[0] = self.body[0][0] + self.direction[1]
            left[1] = self.body[0][1] + self.direction[0]
        else:
            right[0] = self.body[0][0] + self.direction[1]
            right[1] = self.body[0][1] + self.direction[0]
            left[0] = self.body[0][0] - self.direction[1]
            left[1] = self.body[0][1] - self.direction[0]

        front_right = right + self.direction
        front_left = left + self.direction
        back_left = right + -1 * self.direction
        back_right = left + -1 * self.direction

        if front[0] >= self.width or front[0] <= -10 or front[1] >= self.height or front[1] <= -10:
            blocked1 = 2
        elif self.check_value(front):
            blocked1 = 1
        if left[0] >= self.width or left[0] <= -10 or left[1] >= self.height or left[1] <= -10:
            blocked2 = 2
        elif self.check_value(left):
            blocked2 = 1
        if right[0] >= self.width or right[0] <= -10 or right[1] >= self.height or right[1] <= -10:
            blocked3 = 2
        elif self.check_value(right):
            blocked3 = 1
        if front_right[0] >= self.width or front_right[0] <= -10 or right[1] >= self.height or front_right[1] <= -10:
            blocked4 = 2
        elif self.check_value(front_right):
            blocked4 = 1
        if front_left[0] >= self.width or front_left[0] <= -10 or left[1] >= self.height or front_left[1] <= -10:
            blocked5 = 2
        elif self.check_value(front_left):
            blocked5 = 1
        if back_right[0] >= self.width or back_right[0] <= -10 or right[1] >= self.height or back_right[1] <= -10:
            blocked6 = 2
        elif self.check_value(back_right):
            blocked6 = 1
        if back_left[0] >= self.width or back_left[0] <= -10 or left[1] >= self.height or back_left[1] <= -10:
            blocked7 = 2
        elif self.check_value(back_left):
            blocked7 = 1
        return np.array([blocked1, blocked2, blocked3, blocked4, blocked5, blocked6, blocked7])

    def step(self, _action):
        # Traditional Controls
        '''if ((abs(_action - self.previous_action) == 1) and (_action + self.previous_action != 3)):
            self.dead()
        if _action == 0: # Move Up
            self.move([0, -10])
        elif _action == 1: # Move Down
            self.move([0, 10])
        elif _action == 2: # Move Left
            self.move([-10, 0])
        elif _action == 3: # Move Right
            self.move([10, 0])
        self.previous_action = _action'''

        # New Controls
        if _action == 0:    # Move forward
            self.move([self.direction[0], self.direction[1]])
        if self.direction[0] == 0:  # If snake is moving vertically
            if _action == 1:    # Turn Left
                self.move([self.direction[1], self.direction[0]])
                self.direction = [self.direction[1], self.direction[0]]
            if _action == 2:    # Turn Right
                self.move([-self.direction[1], -self.direction[0]])
                self.direction = [-self.direction[1], -self.direction[0]]
        else:   # If snake is moving horizontally
            if _action == 1:    # Turn Left
                self.move([-self.direction[1], -self.direction[0]])
                self.direction = [-self.direction[1], -self.direction[0]]
            if _action == 2:    # Turn Right
                self.move([self.direction[1], self.direction[0]])
                self.direction = [self.direction[1], self.direction[0]]

        if self.check_value(self.body[0]):
            self.dead()
        if self.width in self.body or -10 in self.body:  # Check for wall collisions; works only for square map
            self.dead()
        state = self.get_state()
        return state, self.done, self.reward

    def gen_food(self):
        # Generates Food
        m = False
        while not m:
            self.food = [random.randrange(0, self.width, 10), random.randrange(0, self.height, 10)]
            if not self.check_value(self.food):
                m = True
        self.food_vector = self.body[0] - self.food

    def render(self):
        # Renders the snake, food and the screen
        pygame.draw.rect(self.screen, (0, 0, 0), [self.width, self.height, -self.width, -self.height])
        [pygame.draw.rect(self.screen, (255, 0, 0), [self.body[i][0], self.body[i][1], 10, 10]) for i in range(self.snake_length)]
        pygame.draw.rect(self.screen, (0, 255, 0), [self.food[0], self.food[1], 10, 10])
        pygame.display.flip()

    def open_screen(self):
        # Creates a new screen; Must be called before render()
        self.screen = pygame.display.set_mode(self.size)

    def eat_food(self):
        # Called whenever the snake eats food
        self.snake_length += 1
        self.food_score += 1
        self.reward += 500
        self.hunger = 0
        self.body = np.append(self.body, [self.body[self.snake_length-2] + self.body[self.snake_length-2] - self.body[self.snake_length-3]], axis=0)

    def reset(self):
        # Must be called outside of class
        x = random.randrange(0, self.width, 10)
        y = random.randrange(0, self.width, 10)
        self.body = np.array(np.array([[x, y], [x + 10, y]]))
        self.snake_length = 2
        self.direction = [0, 10]
        self.previous_action = -100
        self.hunger = 0
        self.reward = 0
        self.done = False
        self.food = [random.randrange(0, self.width, 100), random.randrange(0, self.height, 100)]
        self.food_vector = self.body[0] - self.food
        self.food_score = 0
