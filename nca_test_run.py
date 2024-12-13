#%%
import numpy as np
import onnxruntime

class GNCA():
    def __init__(self, height=72, width=72, n_channels=16, model_path=r"./growing-neural-cellular-automata.onnx"):
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
        x[:, self.height // 2 - 10:self.height // 2 + 10, self.width // 2 - 10:self.width // 2 + 10, 3:] = 1.0
        self.input = x
        return x

    def run(self) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: self.input})
        self.output = out[0].astype(np.float32)
        self.input = out[0].astype(np.float32)
        return self.input
# %%
gnca = GNCA()
seed = gnca.make_seeds()
# %%
print(np.shape(seed))
# %%
gnca.run()
# %%
gnca_output = gnca.to_rgba(gnca.input)[0] 
# %%
import pyxel 
# %%

pyxel.init(72, 72)

# %%
old_colors = pyxel.colors.to_list()
# %%
print(len(old_colors))
# %%
pyxel.colors.to_list()
# %%
pyxel.colors[15]
# %%
