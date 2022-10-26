from beamngpy import BeamNGpy, Vehicle
from beamngpy.sensors import Camera


class BeamNGCarCameras:
    def __init__(self, beamng: BeamNGpy, vehicle: Vehicle, training=False):
        direction = (0, -1, 0)
        fov = 70
        resolution = (320, 160)
        y, z = -1.7, 1

        cam_center = Camera(
            bng=beamng,
            vehicle=vehicle,
            name='cam_center',
            pos=(0, y, z),
            dir=direction,
            field_of_view_y=fov,
            resolution=resolution,
            is_render_colours=True,
        )
        if training:
            cam_left = Camera(
                bng=beamng,
                vehicle=vehicle,
                name='cam_left',
                pos=(0.3, y, z),
                dir=direction,
                field_of_view_y=fov,
                resolution=resolution,
                is_render_colours=True
            )
            cam_right = Camera(
                bng=beamng,
                vehicle=vehicle,
                name='cam_right',
                pos=(-0.3, y, z),
                dir=direction,
                field_of_view_y=fov,
                resolution=resolution,
                is_render_colours=True
            )
            self.cameras_array = {"cam_center": cam_center, "cam_left": cam_left, "cam_right": cam_right}
        else:
            self.cameras_array = {"cam_center": cam_center}
