from shapely import geometry


def is_likely_self_intersecting(road_points, lane_width):
    """Verifies two of the self-intersection checks from FreneticV 2022"""
    center_line = geometry.LineString(road_points)

    left_line = center_line.parallel_offset(lane_width, "left")
    right_line = center_line.parallel_offset(lane_width, "right")

    drop_road_checks = {'center_complex': not center_line.is_simple,
                        'left_int_right': left_line.intersects(right_line)}

    return any(drop_road_checks.values())


def is_reframable(map_size, road_points, lane_width):
    """Check if road can be reframed to fit in map"""
    xs = [x for (x, y) in road_points]
    ys = [y for (x, y) in road_points]
    min_xs = min(xs)
    min_ys = min(ys)
    max_xs = max(xs)
    max_ys = max(ys)
    return (max_xs - min_xs <= map_size - 2 * lane_width) and (max_ys - min_ys <= map_size - 2 * lane_width), min_xs, min_ys


def reframe_road(road_points, min_xs, min_ys, lane_width):
    """Road reframing from Frenetic 2021"""
    return [(x - min_xs + lane_width, y - min_ys + lane_width) for (x, y) in road_points]
