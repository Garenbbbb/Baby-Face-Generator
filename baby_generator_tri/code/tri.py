import cv2

# if point is valid
def point_valid(window, p):
    return p[0] > window[0] and p[1] > window[1] and p[0] < window[2] and p[1] < window[3]

def make_tri(width, height, sub_window, d):

    res = []

    tri = sub_window.getTriangleList()

    window = (0, 0, width, height)

    #TODO: int(tri)

    for i in tri :
        pt1 = (int(i[0]), int(i[1]))
        pt2 = (int(i[2]), int(i[3]))
        pt3 = (int(i[4]), int(i[5]))

        if point_valid(window, pt1) and point_valid(window, pt2) and point_valid(window, pt3):
            res.append((d[pt1], d[pt2], d[pt3]))

    return res

def create_tri(width, height, input):

    window = (0, 0, width, height)

    sub_window = cv2.Subdiv2D(window)

    output = []
    for i in input.tolist():
        output.append((int(i[0]),int(i[1])))

    d = {}
    for i in list(zip(output, range(76))):
        d[i[0]] = i[1]

    for p in output:
        sub_window.insert(p)

    return make_tri(width, height, sub_window, d)
