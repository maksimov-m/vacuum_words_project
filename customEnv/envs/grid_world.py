import numpy as np
import pygame
from .bots_agent import AgentBot
import gymnasium as gym
from gymnasium import spaces
import cv2


class help_functions():
    @staticmethod
    def is_in_area(point1, radius1, point2, radius2):
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        return distance < (radius1 + radius2)


WINDOW_WIDTH = 700
WINDOW_HEIGHT = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (0,255,255)


def preprocess_frame(frame):
    # Resize frame to 84x84
    frame = cv2.resize(frame, (84, 84))
    return frame


def count_occurrences(matrix, color):
    # Преобразуем цвет в массив с такой же размерностью, как и у matrix
    color_array = np.array(color)

    # Сравниваем каждую компоненту (RGB) с соответствующими компонентами цвета
    return np.sum(np.all(matrix == color_array, axis=-1))



class Field:
    def __init__(self, agent_number, target_number, agent_radius, target_radius):
        self._agent_number = agent_number
        self._target_number = target_number
        self._agent_radius = agent_radius
        self._target_radius = target_radius
        self._agents = []
        self._agent_targets = []

    def __generate_agent_pos(self):
        # Генерируем случайную позицию для агента
        return [int(np.random.rand(1) * 800), int(np.random.rand(1) * 600)]  # замените 100 на размер вашего поля

    def __generate_target_pos(self):
        # Генерируем случайную позицию для цели
        return [int(np.random.rand(1) * 800), int(np.random.rand(1) * 600)]  # замените 100 на размер вашего поля

    def __generate_agents(self):
        # Генерация агентов, которые не пересекаются
        self._agents = []
        for _ in range(self._agent_number):
            while True:
                new_agent_location = self.__generate_agent_pos()

                # Проверка на пересечение с другими агентами
                if len(self._agents) > 0:
                    overlap = any(
                        help_functions.is_in_area(new_agent_location, self._agent_radius, agent, self._agent_radius)
                        for agent in self._agents)
                else:
                    overlap = False

                if not overlap:
                    self._agents.append(new_agent_location)
                    break

    def __generate_targets_for_agents(self):
        # Генерация целей для каждого агента
        colors = [RED, YELLOW, BLUE]
        self.__agent_targets = []

        for agent_index, agent_location in enumerate(self._agents):
            agent_targets = []

            for _ in range(self._target_number):
                while True:
                    new_target_location = self.__generate_target_pos()

                    # Проверка, что цель не пересекается с агентом
                    if help_functions.is_in_area(new_target_location, self._target_radius, agent_location,
                                                 self._agent_radius):
                        continue

                    # Проверка на пересечение с другими целями данного агента
                    overlap = any(help_functions.is_in_area(new_target_location, self._target_radius, target[:2],
                                                            self._target_radius)
                                  for target in agent_targets)

                    # Проверка на пересечение с целями других агентов
                    if not overlap:
                        overlap = any(help_functions.is_in_area(new_target_location, self._target_radius, target[:2],
                                                                self._target_radius)
                                      for other_agent_targets in self._agent_targets for target in other_agent_targets)

                    if not overlap:
                        agent_targets.append(np.append(new_target_location, colors[agent_index % len(colors)]))
                        break

            self._agent_targets.append(np.array(agent_targets))

    def generate_field(self):
        self.__generate_agents()
        self.__generate_targets_for_agents()


class GridWorld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, world_size=(800, 600), seed=None, targets_number=3):

        self.world_size = world_size # world_size[0] - w, world_size[1] - h


        self.__seed = seed

        self.__agent_radius = 20
        self.__target_radius = 15
        self.__target_number = targets_number

        self.__forward_step = 5
        self.__rotate_angle = 10

        self.__current_angel = 0

        self.bots = []

        self.__agent_location = []
        self.__targets = []  #each element here represented as np array with the following structure [x.pos, y.pos, color]

        self.__canvas = pygame.Surface((self.world_size[0], self.world_size[1]))
        self.__canvas.fill(BLACK)

        self.__target_found = 0  # сколько нашел букв
        # self.area_observed = 0  # количество закрашенных пикселей

        # self.last_steps = []  # последние n шагов
        # self.n_last_steps = 10

        # self.steps_n = 0
        # self.observation_space = spaces.Box(0, 255, shape=(84, 84, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(0, world_size[1] - 1, shape=(8,), dtype=np.float32)

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]),
                                       dtype=np.float32)  # cosX, sinY, (можно добавить power)

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        self.window = None
        self.clock = None

    # def __generate_target_pos(self):
    #     return self.np_random.integers(0 + self.__target_radius, self.world_size - self.__target_radius, size=2,
    #                                    dtype=int)

    # def __generate_agent_location(self):
    #     self.__agent_location = self.np_random.integers(0 + self.__agent_radius,
    #                                                     self.world_size - self.__agent_radius - 1, size=2, dtype=int)

    def __draw_the_head(self, action):
        pygame.draw.line(self.__canvas, RED, self.__agent_location, self.__agent_location + np.array(
            [(self.__agent_radius - 1) * action[0],
             (self.__agent_radius - 1) * action[1]]))

    def reset(self, seed=None, options=None):
        super().reset(seed=self.__seed)  # определяем seed

        print("--------------------")
        # self.__generate_agent_location()
        # self.__generate_targets()
        field = Field(3, self.__target_number, self.__agent_radius, self.__target_radius)

        field.generate_field()

        self.__agent_location = field._agents[0]
        self.__targets = field._agent_targets[0]

        self.bots = [AgentBot(self.__agent_radius, field._agents[i], field._agent_targets[i], self.world_size) for i in range(1,3)]



        self.__target_found = 0
        self.area_observed = 0
        self.__current_angel = 0
        self.steps_n = 0
        self.last_steps = []

        self.__canvas = pygame.Surface((self.world_size[0], self.world_size[1]))
        self.__canvas.fill(BLACK)

        pygame.draw.circle(self.__canvas, WHITE, self.__agent_location, radius=self.__agent_radius)

        #self.__draw_the_head()  # рисуем его направление

        for target in self.__targets:
            pygame.draw.circle(self.__canvas, target[2:], target[:2], radius=self.__target_radius)

        for target_arr in field._agent_targets:
            for target in target_arr:
                pygame.draw.circle(self.__canvas, target[2:], target[:2], radius=self.__target_radius)

        observation = self._get_obs()
        info = self._get_info()  # любая инфа

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        # return preprocess_frame(np.array(pygame.surfarray.pixels3d(self.__canvas)))
        if len(self.__targets) == self.__target_found:
            obs = np.array([*self.__agent_location, *[0, 0], *np.array([self.bots[i]._agent_location for i in range(2)]).ravel()], dtype=np.float32)
        else:
            obs = np.array([*self.__agent_location, *self.__targets[self.__target_found][:2], *np.array([self.bots[i]._agent_location for i in range(2)]).ravel()], dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "collected_goals": self.__target_number - len(self.__targets),
        }

    # def __generate_targets(self):
    #     self.__targets = []
    #
    #     for _ in range(self.__target_number):
    #         while True:
    #             new_target_location = self.__generate_target_pos()
    #
    #             if help_functions.is_in_area(new_target_location, self.__target_radius, self.__agent_location,
    #                                          self.__agent_radius):
    #                 continue
    #
    #             overlap = False
    #             for existing_target_location in self.__targets:
    #                 if help_functions.is_in_area(new_target_location, self.__target_radius,
    #                                              existing_target_location[:2], self.__target_radius):
    #                     overlap = True
    #                     break
    #
    #             if not overlap:
    #                 self.__targets.append(new_target_location)
    #                 break

        # colors = [RED, YELLOW, BLUE]
        # self.__targets = [np.append(pos, color) for pos, color in zip(self.__targets, colors)]
        # self.__targets = np.array(self.__targets)  # Ensure it is a numpy array

    def step(self, action):

        for target in self.__targets:
            pygame.draw.circle(self.__canvas, target[2:], target[:2], radius=self.__target_radius)

        for i in range(2):
            for target in self.bots[i]._targets:
                pygame.draw.circle(self.__canvas, target[2:], target[:2], radius=self.__target_radius)

        [pygame.draw.circle(self.__canvas, BLACK, self.bots[i]._agent_location, self.__agent_radius) for i in range(2)]
        [self.bots[i].update() for i in range(2)]

        done = self.__target_found == self.__target_number

        if done :
            observation = self._get_obs()
            info = self._get_info()
            return observation, 10, done, False, info

        # помечает как посещенное
        pygame.draw.circle(self.__canvas, GREEN, self.__agent_location, self.__agent_radius)

        distance_old = np.linalg.norm(self.__agent_location - self.__targets[self.__target_found][:2])

        # вычисление новой позиции центра агента (нужно передавать угол относительно начала координат)
        direction = np.array([self.__forward_step * action[0],
                              self.__forward_step * action[1]])

        new_loc_x = np.clip(self.__agent_location[0] + direction[0],
                          0 + self.__agent_radius, self.world_size[0] - self.__agent_radius - 1)
        new_loc_y = np.clip(self.__agent_location[1] + direction[1],
                            0 + self.__agent_radius, self.world_size[1] - self.__agent_radius - 1)

        new_loc = [new_loc_x, new_loc_y]
        self.__agent_location = new_loc

        distance_new = np.linalg.norm(self.__agent_location - self.__targets[self.__target_found][:2])

        reward = 0

        dif_dist_1 = np.linalg.norm(np.array(self.__agent_location) - np.array(self.bots[0]._agent_location))
        dif_dist_2 = np.linalg.norm(np.array(self.__agent_location) - np.array(self.bots[1]._agent_location))

        if dif_dist_1 <= (2 * self.__agent_radius) + 0.3:
            reward -= 2

        if dif_dist_2 <= (2 * self.__agent_radius) + 0.3:
            reward -= 2

        if distance_old > distance_new:
            reward += 0.5
        else:
            reward -= 3


        # self.steps_n += 1
        # if self.steps_n % 500 == 0:
        #     reward -= 1

        if help_functions.is_in_area(new_loc, self.__agent_radius, self.__targets[self.__target_found][:2],
                                     self.__target_radius):
            pygame.draw.circle(self.__canvas, GREEN, self.__targets[self.__target_found][:2],
                               self.__target_radius)  # если цель найдена, то закрашиваем в зеленый

            #self.__targets = np.delete(self.__targets, 0, axis=0)  # Remove the first target

            self.__target_found += 1

            reward += 10 * self.__target_found


            print("Нашел цель!")



        # for target in self.__targets[self.__target_found + 1:]:
        #     if help_functions.is_in_area(new_loc, self.__agent_radius, target[:2],
        #                                  self.__target_radius):
        #         reward -= 3

        pygame.draw.circle(self.__canvas, WHITE, self.__agent_location, self.__agent_radius)
        [pygame.draw.circle(self.__canvas, WHITE, self.bots[i]._agent_location, self.__agent_radius) for i in range(2)]

        self.__draw_the_head(action)


        # Заканчиваем игру, если он нашел все буквы



        # # Штрафуем если задел стенку
        # if (self.__agent_location[0] == self.__agent_radius
        #         or self.__agent_location[1] == self.__agent_radius
        #         or (self.world_size - self.__agent_radius - 1) == self.__agent_location[1]
        #         or (self.world_size - self.__agent_radius - 1) == self.__agent_location[0]):
        #     reward -= 1
        #
        # # Награждаем если обследует новую территорию, инач штрафуем
        # if count_occurrences(np.array(pygame.surfarray.pixels3d(self.__canvas)), GREEN) > self.area_observed:
        #     reward += 2
        #
        # else:
        #     reward -= 0.5

        # Обновили обследуемую территорию
        # self.area_observed = count_occurrences(np.array(pygame.surfarray.pixels3d(self.__canvas)), GREEN)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # TODO: добавить условие остановки при проблеме
        # self.last_steps.append(self.__agent_location)
        # if len(self.last_steps) > self.n_last_steps:
        #     self.last_steps.pop(0)

        # end_game = ((abs(sum([x[0] for x in self.last_steps]) / len(self.last_steps) - self.last_steps[0][0]) < 0.7
        #                  and abs(sum([x[1] for x in self.last_steps]) / len(self.last_steps) - self.last_steps[0][1]) < 0.7)
        #                  and len(self.last_steps) == self.n_last_steps)

        return observation, reward, done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.world_size[0], self.world_size[1]))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.window.blit(self.__canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.array(pygame.surfarray.pixels3d(self.__canvas))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
