from descartes import PolygonPatch
import numpy as np
import matplotlib.pyplot as plt
import typing
import os
from shapely.geometry import LineString

from ambiegenvae.generators.abstract_generator import AbstractGenerator

from ambiegenvae.common.road_validity_check import is_valid_road

from ambiegenvae.common.car_road import Map
from ambiegenvae.common.vehicle_evaluate import interpolate_road
from numpy import dot
from numpy.linalg import norm

class AmbieGenRoadGenerator(AbstractGenerator):
    """
    Class to generate a road based on a kappa function.
    Part of the code is based on the following repository:

    Args:
        map_size (int): The size of the map.
        solution_size (int): The size of the solution.

    Attributes:
        min_len (int): The minimum length of the road.
        max_len (int): The maximum length of the road.
        min_angle (int): The minimum angle of the road.
        max_angle (int): The maximum angle of the road.
        map_size (int): The size of the map.
        road_points (list): The points on the road.
        scenario (list): The scenario of the road.

    Properties:
        phenotype_size (int): The size of the phenotype.
        genotype (list): The phenotype of the generator.

    Methods:
        set_genotype(genotype: list): Set the phenotype of the generator.
        genotype2phenotype(genotype: list) -> tuple: Convert the genotype to phenotype.
        cmp_func(x, y): Compare two values using cosine similarity.
        get_phenotype() -> tuple: Get the genotype of the generator.
        generate_random_test() -> tuple: Generate a random test.
        visualize_test(road_points: np.ndarray, save_path: str = "test", num: int = 0, title: str = ""): Visualize the test.

    """

    def __init__(self, map_size: int, solution_size: int):
        super().__init__(solution_size)

        self.min_len = 5
        self.max_len = 30
        self.min_angle = 10
        self.max_angle = 90
        self.map_size = map_size
        self.road_poiunts = []
        self.scenario = []

    @property
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size  # max_number_of_points

    @property
    def genotype(self) -> typing.List[float]:
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        return self.scenario

    def set_genotype(self, genotype: typing.List[float]):
        """Set the phenotype of the generator.

        Args:
            genotype (list): The phenotype of the generator.
        """
        self.scenario = genotype

    def genotype2phenotype(
        self, genotype: typing.List[float]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Convert the genotype to phenotype.

        Args:
            genotype (list): The genotype of the generator.

        Returns:
            tuple: The phenotype of the generator.
        """
        self.set_genotype(genotype)
        phenotype = self.get_phenotype()
        return phenotype

    def cmp_func(self, x, y):
        """Compare two values using cosine similarity.

        Args:
            x: The first value.
            y: The second value.

        Returns:
            float: The difference between the two values.
        """
        cos_sim = dot(x, y) / (norm(x) * norm(y))

        difference = 1 - abs(cos_sim)
        return difference

    def get_phenotype(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the genotype of the generator.

        Returns:
            tuple: The genotype of the generator.
        """
        map = Map(self.map_size)
        road_points, scenario = map.get_points_from_states(self.genotype)

        return road_points

    def generate_random_test(self) -> (typing.List[float], bool):
        """Generate a random test.

        Returns:
            tuple: The road points and a boolean indicating if the road is valid.
        """
        actions = list(range(0, 3))
        lengths = list(range(self.min_len, self.max_len))
        angles = list(range(self.min_angle, self.max_angle))

        map_size = self.map_size

        done = False
        test_map = Map(map_size)
        while not done:
            action = np.random.choice(actions)
            if action == 0:
                length = np.random.choice(lengths)
                done = not (test_map.go_straight(length))
            elif action == 1:
                angle = np.random.choice(angles)
                done = not (test_map.turn_right(angle))
            elif action == 2:
                angle = np.random.choice(angles)
                done = not (test_map.turn_left(angle))
        scenario = test_map.scenario

        map = Map(map_size)
        road_points, scenario = map.get_points_from_states(scenario)
        road_points = interpolate_road(road_points)

        self.road_poiunts = road_points
        self.scenario = scenario

        valid = is_valid_road(road_points, map_size)

        return road_points, valid

    def visualize_test(
        self,
        road_points: np.ndarray,
        save_path: str = "test",
        num: int = 0,
        title: str = "",
    ):
        """Visualize the test.

        Args:
            road_points (np.ndarray): The road points.
            save_path (str, optional): The path to save the image to. Defaults to "test".
            num (int, optional): The number of the image. Defaults to 0.
            title (str, optional): The title of the image. Defaults to "".
        """
        road_points = list(road_points)

        intp_points = road_points

        fig, ax = plt.subplots(figsize=(8, 8))
        road_x = []
        road_y = []

        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        top = self.map_size
        bottom = 0

        road_line = LineString(road_points)
        ax.plot(road_x, road_y, "yo--", label="Road")

        road_poly = LineString([(t[0], t[1]) for t in intp_points]).buffer(
            4.0, cap_style=2, join_style=2
        )
        road_patch = PolygonPatch(
            (road_poly), fc="gray", ec="dimgray"
        )
        ax.add_patch(road_patch)

        ax.set_xlim(road_line.bounds[0], road_line.bounds[2])
        ax.set_ylim(road_line.bounds[1], road_line.bounds[3])

        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.legend(fontsize=16)
        ax.set_ylim(bottom, top)
        plt.ioff()
        ax.set_xlim(bottom, top)
        ax.set_title(title, fontsize=16)

        ax.legend()
        if not (os.path.exists(save_path)):
            os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path + "\\" + str(num) + ".png", bbox_inches="tight")
        plt.close(fig)
