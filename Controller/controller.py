import math

class EntropyGapController:
    def __init__(self, gap_small=0.02, gap_medium=0.08, slope_good=-0.02):
        self.gap_small = gap_small
        self.gap_medium = gap_medium
        self.slope_good = slope_good

    def choose_factor(self, H, th, num_classes, H_prev=None):
        
        Hmax = math.log(num_classes)  # natural log if your entropy uses ln
        g = (H - th) / max(Hmax, 1e-9)

        if H_prev is not None:
            sH = H - H_prev
            if sH >= 0:
                return 1

        if g <= self.gap_small:
            return 64
        elif g <= self.gap_medium:
            return 8
        else:
            return 1
