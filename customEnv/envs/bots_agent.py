import numpy as np

class help_functions():
    @staticmethod
    def is_in_area(point1, radius1, point2, radius2):
        distance = np.linalg.norm(np.array(point1) - np.array(point2))
        return distance < (radius1 + radius2)


class AgentBot:
    def __init__(self, agent_radius, agent_location, targets, world_size):
        self._world_size = world_size
        self._agent_radius = agent_radius
        self._current_angle = 0  # Текущий угол в градусах
        self._target_radius = 15
        self._forward_step = 5  # Шаг вперед
        self._rotate_angle = 10  # Угол поворота
        self._agent_location = np.array(agent_location)  # Положение агента
        self._targets = [np.array(target) for target in targets]  # Список целей
        self._target_found = 0  # Счетчик достигнутых целей

    def update(self, arr_loc):
        if not self._targets:
            return  # Если целей нет, ничего не делаем

        # Цель для достижения
        target_location = self._targets[0]

        # Вектор к цели
        direction_vector = target_location[:2] - self._agent_location
        target_angle = np.degrees(np.arctan2(direction_vector[1], direction_vector[0]))

        # Добавляем случайный шум к углу цели
        noise = np.random.uniform(-70, 70)  # Измените диапазон для большего или меньшего шума
        target_angle += noise

        # Вычисление разницы углов
        angle_difference = target_angle - self._current_angle
        angle_difference = (angle_difference + 180) % 360 - 180  # Нормализация разницы углов до [-180, 180]

        # Поворот на нужный угол
        if abs(angle_difference) > self._rotate_angle:
            self._current_angle += self._rotate_angle * np.sign(angle_difference)
        else:
            self._current_angle += angle_difference  # Точный поворот, если угол малый

        # Обновляем текущий угол в пределах [0, 360)
        self._current_angle %= 360

        # Шаг вперед
        angle_rad = np.radians(self._current_angle)
        step_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * self._forward_step
        new_loc_x = np.clip(self._agent_location[0] + step_vector[0],
                            0 + self._agent_radius, self._world_size[0] - self._agent_radius - 1)
        new_loc_y = np.clip(self._agent_location[1] + step_vector[1],
                            0 + self._agent_radius, self._world_size[1] - self._agent_radius - 1)

        new_loc = [new_loc_x, new_loc_y]

        # Проверка на столкновение с другими агентами
        for enemy_loc in arr_loc:
            if help_functions.is_in_area(new_loc, self._agent_radius, enemy_loc[:2], self._target_radius):
                new_loc = self._agent_location  # Оставляем агента на старой позиции при столкновении

        # Обновляем позицию агента
        self._agent_location = new_loc

        # Проверка, достигнута ли цель
        if np.linalg.norm(self._agent_location - target_location[:2]) <= self._agent_radius:
            self._target_found += 1
            self._target_found %= 3
            self._targets.append(self._targets.pop(0))

