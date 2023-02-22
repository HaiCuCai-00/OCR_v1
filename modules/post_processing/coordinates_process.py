def get_area(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


def get_inter_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea


def check_point_in_box(point, box):
    if (
        box[0] < point[0]
        and point[0] < box[2]
        and box[1] < point[1]
        and point[1] < box[3]
    ):
        return True
    else:
        return False


def get_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_overlap(y1, y2, y3, y4):
    """
    get overlap between y1y2 and y3y4
    """
    return max(0, min(y2, y4) - max(y1, y3))


def check_is_same_row(box1, box2, overlap_x_thresh=0.2, overlap_y_thresh=0.33):
    overlap_x = get_overlap(box1[0][0], box1[1][0], box2[0][0], box2[1][0])
    if (
        overlap_x / float(box1[1][0] - box1[0][0]) > overlap_x_thresh
        or overlap_x / float(box2[1][0] - box2[0][0]) > overlap_x_thresh
    ):
        return False
    overlap_y = get_overlap(box1[1][1], box1[2][1], box2[0][1], box2[3][1])
    if (
        overlap_y / float(box1[2][1] - box1[1][1]) > overlap_y_thresh
        and overlap_y / float(box2[2][1] - box2[1][1]) > overlap_y_thresh
    ):
        return True
    else:
        return False


def sort_box_in_line(boxes, line):
    line = sorted(line, key=lambda x: boxes[x][0][0])
    return line


def sort_line(boxes, lines):
    lines = sorted(lines, key=lambda x: boxes[x[0]][0][1])
    return lines


def find_lines(boxes):
    lines = []
    for i, box in enumerate(boxes):
        not_found = True
        for line in lines:
            for line_box_idx in line:
                if not_found and check_is_same_row(box, boxes[line_box_idx]):
                    line.append(i)
                    not_found = False
        if not_found:
            lines.append([i])

    sort_lines = []
    for line in lines:
        sort_lines.append(sort_box_in_line(boxes, line))

    sort_lines = sort_line(boxes, sort_lines)

    return sort_lines
