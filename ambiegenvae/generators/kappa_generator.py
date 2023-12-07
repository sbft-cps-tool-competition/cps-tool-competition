import time
import typing
import random
import logging  # as log
from shapely.geometry import LineString
from descartes import PolygonPatch
import numpy as np
import matplotlib.pyplot as plt
import os

from ambiegenvae.generators.abstract_generator import AbstractGenerator

from ambiegenvae.common.road_validity_check import is_valid_road

from ambiegenvae.common.road_validity_check import min_radius

from numpy import dot
from numpy.linalg import norm

log = logging.getLogger(__name__)
MAX_RADIUS_THRESHOLD = 130



class KappaRoadGenerator(AbstractGenerator):
    """
    Class to generate a road based on a kappa function.
    Part of the code is based on the following repository:
    https://github.com/ERATOMMSD/frenetic-lib/tree/main

    Args:
        map_size (int): The size of the map.
        solution_size (int, optional): The size of the solution. Defaults to 25.
        theta0 (float, optional): The initial angle of the line. Defaults to 1.57.
        segment_length (float, optional): The length of each segment. Defaults to 10.
        margin (float, optional): The margin of the road. Defaults to 8.

    Attributes:
        global_bound (float): The global bound for kappa values.
        local_bound (float): The local bound for kappa values.
        theta0 (float): The initial angle of the line.
        segment_length (float): The length of each segment.
        margin (float): The margin of the road.
        map_offset (int): The offset of the map.
        kappas (list): The list of kappa values.
        map_size (int): The size of the map.
        _name (str): The name of the generator.

    Properties:
        phenotype_size (int): The size of the phenotype.
        genotype (list): The phenotype of the generator.

    Methods:
        set_genotype(genotype: list): Sets the phenotype of the generator.
        get_phenotype() -> tuple: Gets the genotype of the generator.
        genotype2phenotype(genotype: list) -> tuple: Converts the genotype to phenotype.
        cmp_func(x, y): Computes the difference between two vectors.
        get_next_kappa(last_kappa: float) -> float: Generates a new kappa value based on the previous value.
        generate_random_kappas() -> list: Generates a list of random kappa values.
        reframe_road(xs, ys) -> tuple: Reframes the road to fit the map size.
        frenet_to_cartesian(x0: float, y0: float, theta0: float, ss: np.ndarray, kappas: list) -> tuple: Converts Frenet points to Cartesian coordinates.
        kappas_to_road_points(kappas: list) -> np.array: Converts kappa values to road points in Cartesian coordinates.
        visualize_test(road_points: np.ndarray, save_path: str = "test", num: int = 0, title: str = ""): Visualizes the road and car path.
        generate_random_test() -> tuple: Generates a road using the kappa function.

    """

    def __init__(
        self,
        map_size: int,
        solution_size: int = 25,
        theta0: float = 1.57,
        segment_length: float = 10,
        margin: float = 8,
    ):

        super().__init__(solution_size)

        self.global_bound = 0.07  # 0.06232
        self.local_bound = 0.05
        self.theta0 = theta0
        self.segment_length = segment_length
        self.margin = margin
        self.map_offset = 5
        self.kappas = []
        self.map_size = map_size
        self._name = "KappaRoadGenerator"

    @property
    def phenotype_size(self) -> int:
        """Size of the phenotype.

        Returns:
            int: Size of the phenotype.
        """
        return self.size  

    @property
    def genotype(self) -> typing.List[float]:
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        return self.kappas

    def set_genotype(self, genotype: typing.List[float]):
        """Phenotype of the generator.

        Returns:
            list: Phenotype of the generator.
        """
        self.kappas = genotype

    def get_phenotype(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Get the genotype of the generator.

        Returns:
            list: Genotype of the generator.
        """
        road_points = self.kappas_to_road_points(self.kappas)
        return road_points

    def genotype2phenotype(
        self, genotype: typing.List[float]
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        self.set_genotype(genotype)
        phenotype = self.get_phenotype()
        return phenotype

    def cmp_func(self, x, y):
        cos_sim = dot(x, y) / (norm(x) * norm(y))

        difference = 1 - abs(cos_sim)
        return difference

    def get_next_kappa(self, last_kappa: float) -> float:
        """
        Generates a new kappa value based on the previous value.

        Args:
            previous_kappa: the previous kappa value

        Returns:
            a new kappa value
        """
        return random.choice(
            np.linspace(
                max(-self.global_bound, last_kappa - self.local_bound),
                min(self.global_bound, last_kappa + self.local_bound),
            )
        )

    def generate_random_kappas(self) -> typing.List[float]:
        """
        Generates a test using frenet framework to determine the curvature of the points.

        Returns:
            a list of kappa values and its cartesian representation.
        """
        points = self.size  # + random.randint(-5, 5)
        # Producing randomly generated kappas for the given setting.
        kappas = ([0.0]) * points
        for i in range(len(kappas)):
            kappas[i] = self.get_next_kappa(kappas[i - 1])

        self.kappas = kappas

        return kappas

    def reframe_road(self, xs, ys):
        """
        Args:
            xs: cartesian x coordinates
            ys: cartesian y coordinates
        Returns:
            A representation of the road that fits the map size (when possible).
        """
        min_xs = min(xs)
        min_ys = min(ys)
        road_width = self.margin  # TODO: How to get the exact road width?
        if (max(xs) - min_xs + road_width > self.map_size - self.margin) or (
            max(ys) - min_ys + road_width > self.map_size - self.margin
        ):
            log.info("Skip: Road won't fit")
            return (xs[:-2], ys[:-2])  # np.array([]), np.array([])
            # TODO: Fail the entire test and start over
        xs = list(map(lambda x: x - min_xs + road_width, xs))
        ys = list(map(lambda y: y - min_ys + road_width, ys))
        return (xs, ys)  # list(zip(xs, ys))

    def frenet_to_cartesian(
        self,
        x0: float,
        y0: float,
        theta0: float,
        ss: np.ndarray,
        kappas: typing.List[float],
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Trapezoidal integration to compute Cartesian coordinates from given curvature values."""
        xs = np.zeros(len(kappas))
        ys = np.zeros(len(kappas))
        thetas = np.zeros(len(kappas))
        xs[0] = x0
        ys[0] = y0
        thetas[0] = theta0
        for i in range(thetas.shape[0] - 1):
            ss_diff_half = (ss[i + 1] - ss[i]) / 2.0
            thetas[i + 1] = thetas[i] + (kappas[i + 1] + kappas[i]) * ss_diff_half
            xs[i + 1] = (
                xs[i] + (np.cos(thetas[i + 1]) + np.cos(thetas[i])) * ss_diff_half
            )
            ys[i + 1] = (
                ys[i] + (np.sin(thetas[i + 1]) + np.sin(thetas[i])) * ss_diff_half
            )
        (xs, ys) = self.reframe_road(xs, ys)
        return (xs, ys)

    def kappas_to_road_points(self, kappas: typing.List[float]) -> np.array:
        """
        Args:
            kappas: list of kappa values
            frenet_step: The distance between to points.
            theta0: The initial angle of the line. (1.57 == 90 degrees)
        Returns:
            road points in cartesian coordinates
        """
        # Using the bottom center of the map.
        y0 = self.map_size / 2  # self.margin
        x0 = self.map_size / 2
        theta0 = self.theta0
        ss = np.cumsum([self.segment_length] * len(kappas)) - self.segment_length
        # Transforming the frenet points to cartesian
        (xs, ys) = self.frenet_to_cartesian(x0, y0, theta0, ss, kappas)

        return np.column_stack([xs, ys])

    def visualize_test(
        self,
        road_points: np.ndarray,
        save_path: str = "test",
        num: int = 0,
        title: str = "",
    ):
        """
        It takes a list of states, and plots the road and the car path

        Args:
          states: a list of tuples, each tuple is a state of the car.
          save_path: The path to save the image to. Defaults to test.png
        """
        road_points = list(road_points)

        intp_points = road_points  # interpolate_road(road_points)

        fig, ax = plt.subplots(figsize=(8, 8))
        road_x = []
        road_y = []

        for p in intp_points:
            road_x.append(p[0])
            road_y.append(p[1])

        top = self.map_size
        bottom = 0

        road_line = LineString(road_points)
        ax.plot(road_x[0], road_y[0], "ro", label="Start")
        ax.plot(road_x[1:], road_y[1:], "yo--", label="Road")
        # Plot the road as a line with custom styling
        # ax.plot(*road_line.xy, color='gray', linewidth=10.0, solid_capstyle='round', zorder=4)
        road_poly = LineString([(t[0], t[1]) for t in intp_points]).buffer(
            4.0, cap_style=2, join_style=2
        )
        road_patch = PolygonPatch(
            (road_poly), fc="gray", ec="dimgray"
        )  # ec='#555555', alpha=0.5, zorder=4)
        ax.add_patch(road_patch)

        # Set axis limits to show the entire road
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

    def generate_random_test(self):
        """
        Generates a road using the kappa function.

        Returns:
            a list of road points.
        """
        self.kappas = self.generate_random_kappas()
        road_points = self.kappas_to_road_points(self.kappas)

        valid_1, _ = is_valid_road(
            road_points, map_size=self.map_size, consider_bounds=True
        )
        valid_2 = min_radius(road_points) < MAX_RADIUS_THRESHOLD
        valid = valid_1 and valid_2

        return road_points, valid


if __name__ == "__main__":
    gen = KappaRoadGenerator(200)
    start = time.time()
    road = gen.generate_random_test()
    print("Gen_time", time.time() - start)
    gen.visualize_test(road)
    print(road)
    print(gen.kappas)
