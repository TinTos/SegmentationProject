class RectangleLabeler:
    def __init__(self, overview, rectangles, labels, tilesize, shiftcount):
        self.overview = overview
        self.rectangles = rectangles
        self.tilesize = tilesize
        self.labels = labels
        self.tiles = self.get_tiles()
        self.shiftcount = shiftcount

    def order_rectangle(self, rectangle):
        x = [r for r in rectangle[1]]
        y = [r for r in rectangle[0]]

        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)

        return (miny, minx), (maxy, maxx)

    def get_tiles(self):
        result = []
        count = 0
        for r in self.rectangles:
            ro = self.order_rectangle(r)
            mintx = ro[0][1] // self.tilesize + (1 if ro[0][1] % self.tilesize != 0 else 0)
            minty = ro[0][0] // self.tilesize + (1 if ro[0][0] % self.tilesize != 0 else 0)

            maxtx = (ro[1][1] - self.tilesize) // self.tilesize - 1
            maxty = (ro[1][0] - self.tilesize) // self.tilesize - 1

            result.append((minty, mintx), (maxty, maxtx), self.labels[count])

            count += 1


        return result








