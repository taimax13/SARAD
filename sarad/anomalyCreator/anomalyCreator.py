import random

class AnomalyCreator:

    def __init__(self):
        pass

    def add_stripes(self, img):
        damaged = img.copy()
        for i in range(0, damaged.shape[0], 10):
            damaged[i:i + 2, :, :] = 255
        return damaged

    def add_dots(self, img):
        damaged = img.copy()
        h, w, _ = damaged.shape
        for _ in range(200):
            x = random.randint(0, w - 2)
            y = random.randint(0, h - 2)
            damaged[y:y + 2, x:x + 2, :] = 0
        return damaged

    def add_noise(self, img, mean=0, std=25):
        noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
        return np.clip(img + noise, 0, 255)

    def invert_patch(self,img):
        damaged = img.copy()
        h, w, _ = damaged.shape
        x, y = random.randint(0, w // 2), random.randint(0, h // 2)
        damaged[y:y + 50, x:x + 50, :] = 255 - damaged[y:y + 50, x:x + 50, :]
        return damaged

    def random_mask(self,img):
        damaged = img.copy()
        h, w, _ = damaged.shape
        for _ in range(3):
            x = random.randint(0, w - 30)
            y = random.randint(0, h - 30)
            damaged[y:y + 20, x:x + 20, :] = np.random.randint(0, 256, (20, 20, 3))
        return damaged

    def apply_random_damage(self, img):
        funcs = [self.add_stripes, self.add_dots, self.add_noise, self.invert_patch, self.random_mask]
        return random.choice(funcs)(img)