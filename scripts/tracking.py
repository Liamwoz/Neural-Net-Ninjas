from collections import defaultdict


class PlayerTrackingHistory:
    def __init__(self, max_history=20):
        self.history = defaultdict(list)
        self.max_history = max_history

    def update(self, track_id, box):
        x1, y1, x2, y2 = box

        # Bottom-center = approximate position on pitch
        x = float((x1 + x2) / 2)
        y = float(y2)

        self.history[track_id].append((x, y))

        if len(self.history[track_id]) > self.max_history:
            self.history[track_id].pop(0)

        return x, y, self.history[track_id]