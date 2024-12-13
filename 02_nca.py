import pyxel
import numpy as np
import onnxruntime

class GNCA():
    def __init__(self, height=72, width=72, n_channels=16, model_path=r"./gray_scott.onnx"):
        self.height: int = height
        self.width: int = width
        self.n_channels: int = n_channels
        self.session = onnxruntime.InferenceSession(model_path)
        self.input: np.ndarray
        self.output: np.ndarray
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    def to_alpha(self, x):
        return np.clip(x[..., 3:4], 0, 0.9999)

    def to_rgba(self, x):
        rgb, a = x[..., :3], self.to_alpha(x)
        return np.concatenate((np.clip(1.0 - a + rgb, 0, 0.9999), a), axis=3)

    def write_alpha_tolist(self, x):
        alpha = self.to_alpha(x).tolist()
        return alpha

    def make_seeds(self):
        x = np.zeros([1, self.height, self.width, self.n_channels], np.float32)
        x[:, self.height // 2 - 1:self.height // 2 + 1, self.width // 2 - 1:self.width // 2 + 1, 3:] = 1.0
        self.input = x
        return x

    def run(self) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: self.input})
        self.output = out[0].astype(np.float32)
        self.input = out[0].astype(np.float32)
        return self.input

class App():
    def __init__(self):
        pyxel.init(60,60)
        self.gnca = GNCA(height=60, width=60)
        self.gnca.make_seeds()
        pyxel.run(self.update,self.draw)

    def closest_color_index(self, r, g, b):
        min_dist = float('inf')
        best_idx = 0
        for i, color_val in enumerate(pyxel.colors.to_list()):
            pr = (color_val >> 16) & 0xFF
            pg = (color_val >> 8) & 0xFF
            pb = color_val & 0xFF
            dr = pr - r
            dg = pg - g
            db = pb - b
            dist = dr*dr + dg*dg + db*db
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        return best_idx

    def update(self):
        self.gnca.run()
    def draw(self):
        gnca_output = self.gnca.to_rgba(self.gnca.input)[0]
        # -> (72, 72, 4)
        # we need to create some function that convert RGB into 24bit
        for row in range(60):
            for col in range(60):
                r, g, b = [int(x*255) for x in gnca_output[row, col, :3]]
                #pyxel.colors[15] = int((r << 16) + (g << 8) + b)
                c = self.closest_color_index(r,g,b)
                
                pyxel.pset(row, col, c)
        self.gnca.input[0, pyxel.mouse_x-4:pyxel.mouse_x+4,pyxel.mouse_y-4:pyxel.mouse_y+4,3] = 0 

App()