import pyxel
import numpy as np
import onnxruntime
import random

bullets = []
enemy_seeds = []
gncas = []
BULLET_SPEED = 3
x = 24
y = 10

def update_entities(entities):
    for entity in entities:
        entity.update()

def cleanup_entities(entities):
    for i in range(len(entities) - 1, -1, -1):
        if not entities[i].is_alive:
            del entities[i]

def draw_entities(entities):
    for entity in entities:
        entity.draw()

class GNCA():
    def __init__(self,position_x, position_y, height=72, width=72, n_channels=16, model_path=r"./growing-neural-cellular-automata.onnx"):
        self.height: int = height
        self.width: int = width
        self.x = position_x
        self.y = position_y
        self.n_channels: int = n_channels
        self.session = onnxruntime.InferenceSession(model_path)
        self.input: np.ndarray
        self.output: np.ndarray
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.make_seeds()
        gncas.append(self)
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
        self.run()

    def draw(self):
        gnca_output = self.to_rgba(self.input)[0]
        for row in range(72):
            for col in range(72):
                r, g, b = [int(x*255) for x in gnca_output[row, col, :3]]
                #pyxel.colors[15] = int((r << 16) + (g << 8) + b)
                c = self.closest_color_index(r,g,b)
                if c == 7:
                    pass
                else:
                    pyxel.pset(row+self.x, col+self.y, c)

class Player():
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def update(self):
        if pyxel.btnp(pyxel.KEY_LEFT):
            self.x -= 8
        elif pyxel.btnp(pyxel.KEY_RIGHT):
            self.x += 8
        elif pyxel.btnp(pyxel.KEY_UP):
            self.y -= 8
        elif pyxel.btnp(pyxel.KEY_DOWN):
            self.y += 8
        if pyxel.btnp(pyxel.KEY_SPACE):
            Ballet(self.x, self.y)
        
        if self.x >= 120-15:
            self.x =120-15
        if self.y >= 120-15:
            self.y = 120-15
        if self.x <= 0:
            self.x = 0
        if self.y <= 0:
            self.y = 0
    def draw(self):
        pyxel.blt(self.x, self.y, 0, 0, 24, 15, 15,0) 

class Ballet():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.w = 8
        self.h = 8
        self.is_alive = True
        bullets.append(self)
    def update(self):
        self.y -= BULLET_SPEED
        if self.y + self.h - 1 < 0:
            self.is_alive = False
    def draw(self):
        pyxel.blt(self.x, self.y,0, 8, 16, self.w, self.h,0)

class Enemmy_seed():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.h = 8
        self.w = 8
        self.is_alive = True
        enemy_seeds.append(self)
    def update(self):
        self.x += random.randint(-4,4)
        self.y += random.randint(-4,4)
        if self.y + self.h - 1 < 0 or self.x + self.x < 0 or self.x-self.h>120 or self.y - self.h > 120 or random.random() > 0.96:
            self.is_alive = False
    def draw(self):
        pyxel.blt(self.x, self.y,0, 16, 16, self.w, self.h, 0)
        
class App():
    def __init__(self):
        pyxel.init(120,120)
        pyxel.load("my_resource.pyxres")
        self.player = Player(60,100)
        self.gnca = GNCA(position_x = x, position_y = y)
        pyxel.run(self.update,self.draw)

    def update(self):
        self.update_play_scene()
    
    def draw(self):
        pyxel.cls(1)
        self.draw_play_scene()

    def update_play_scene(self):
        for bullet in bullets:
            if (bullet.x >= 24) and (bullet.x <= 96) and (bullet.y >=10) and (bullet.y) <= 82 :
                self.gnca.input[0, bullet.x-4-x:bullet.x+4-x,bullet.y-4-y:bullet.y+4-y,3] = 0
                if random.random() > 0.95:
                    Enemmy_seed(bullet.x, bullet.y)
        self.player.update()
        self.gnca.update()
        update_entities(bullets)
        update_entities(enemy_seeds)
        cleanup_entities(bullets)
        cleanup_entities(enemy_seeds)

    
    def draw_play_scene(self):
        self.player.draw()
        self.gnca.draw()
        draw_entities(bullets)
        draw_entities(enemy_seeds)

        
App()