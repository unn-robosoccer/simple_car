import cv2
from utils import get_edges, linedetect


def vision(frame, show=False, center_point=None):
    if show:
        cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

    edges = get_edges(frame)
    result = linedetect(edges, center_point)

    if show:
        image_with_lines = result['image']
        image_with_lines = cv2.addWeighted(frame, 1, image_with_lines, 0.8, 0)
        cv2.imshow('detected road', cv2.resize(image_with_lines, (0, 0), fx=0.5, fy=0.5))

    state = result['distance_to_center']
    return state
