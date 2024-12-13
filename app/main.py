import asyncio
import pyodide_js
import random

await pyodide_js.loadPackage('onnxruntime')
await pyodide_js.loadPackage('numpy')

import onnxruntime
import numpy as np

SCENE_TITLE = 0
SCENE_PLAY = 1
SCENE_GAMEOVER = 2

bullets = []
enemy_seeds = []
gncas = []
BULLET_SPEED = 3
MAX_LIFE = 1000
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

def closest_color_index(r, g, b):
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

def draw_nca(researd, grayscott):
    for row in range(72):
        for col in range(72):
            # Draw researd
            r, g, b = [int(x*255) for x in researd[row, col, :3]]
            c = closest_color_index(r,g,b)
            if c != 7:
                pyxel.pset(row+x, col+y, c)
            
            # Draw grayscott (using scaled coordinates)
            if row < 60 and col < 60:
                r, g, b = [int(x*255) for x in grayscott[row, col, :3]]
                c = closest_color_index(r,g,b)
                if c != 7:
                    pyxel.pset(row*2, col*2, c)

class GS():
    def __init__(self,position_x, position_y, height=60, width=60, n_channels=16, model_path=r"./gray_scott_3.onnx"):
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
        self.initialize()
        gncas.append(self)
    def to_alpha(self, x):
        return np.clip(x[..., 3:4], 0, 0.9999)

    def to_rgba(self, x):
        rgb, a = x[..., :3], self.to_alpha(x)
        return np.concatenate((np.clip(1.0 - a + rgb, 0, 0.9999), a), axis=3)

    def write_alpha_tolist(self, x):
        alpha = self.to_alpha(x).tolist()
        return alpha

    def initialize(self):
        x = np.zeros([1, self.height, self.width, self.n_channels], np.float32)
        #x[:, self.height // 2 - 1:self.height // 2 + 1, self.width // 2 - 1:self.width // 2 + 1, 3:] = 1.0
        self.input = x
        return x
    
    def run(self) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: self.input})
        self.output = out[0].astype(np.float32)
        self.input = out[0].astype(np.float32)
        return self.input
    
    def update(self):
        self.run()
    def draw(self):
        return self.to_rgba(self.input)[0]

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
    
    def update(self):
        self.run()
    def draw(self):
        return self.to_rgba(self.input)[0]
    
class Player():
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.life = MAX_LIFE
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
        self.damage_count = 0
        self.damage_threshold = 200
        self.is_alive = True
        bullets.append(self)
    def update(self):
        self.y -= BULLET_SPEED
        if self.y + self.h - 1 < 0:
            self.is_alive = False
    def draw(self):
        pyxel.blt(self.x, self.y,0, 8, 16, self.w, self.h,0)

class Enemy_seed():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.h = 8
        self.w = 8
        self.is_alive = True
        enemy_seeds.append(self)
    def update(self):
        self.x += random.randint(-8,8)
        self.y += random.randint(-8,8)
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
        self.gs = GS(position_x = 0, position_y = 0)
        self.scene = SCENE_TITLE
        pyxel.run(self.update,self.draw)

    def update(self):
        if self.scene == SCENE_TITLE:
            self.update_title_scene()
        elif self.scene == SCENE_PLAY:
            self.update_play_scene()
        elif self.scene == SCENE_GAMEOVER:
            self.update_gameover_scene()
    
    def draw(self):
        pyxel.cls(1)
        if self.scene == SCENE_TITLE:
            self.draw_title_scene()
        elif self.scene == SCENE_PLAY:
            self.draw_play_scene()
        elif self.scene == SCENE_GAMEOVER:
            self.draw_gameover_scene()
    
    def update_title_scene(self):
        if pyxel.btnp(pyxel.KEY_RETURN):
            self.scene = SCENE_PLAY
        
    def update_play_scene(self):
        for bullet in bullets:
            if (bullet.x >= 24) and (bullet.x <= 96) and (bullet.y >=10) and (bullet.y) <= 82 :
                gnca_region = self.gnca.input[0, bullet.x-4-x:bullet.x+4-x, bullet.y-4-y:bullet.y+4-y, 3]
                gnca_nonzero_count = np.count_nonzero(gnca_region)
                bullet.damage_count += gnca_nonzero_count
                self.gnca.input[0, bullet.x-4-x:bullet.x+4-x, bullet.y-4-y:bullet.y+4-y, 3] = 0
                
                if random.random() > 0.95:
                    Enemy_seed(bullet.x, bullet.y)
                    
            gs_region = self.gs.input[0, bullet.x//2-4:bullet.x//2+4,bullet.y//2-4:bullet.y//2+4,3]
            gs_nonzero_count = np.count_nonzero(gs_region)
            bullet.damage_count += gs_nonzero_count
            self.gs.input[0, bullet.x//2-4:bullet.x//2+4,bullet.y//2-4:bullet.y//2+4,3] = 0

            if bullet.damage_count > bullet.damage_threshold:
                bullet.is_alive = False
        
        for seed in enemy_seeds:
            if random.random() > 0.97:
                self.gs.input[0, seed.x//2 -2:seed.x//2+2, seed.y//2-2:seed.y//2+2, 3] = 1
        
        player_region = self.gs.input[0, self.player.x//2-4:self.player.x//2+4,self.player.y//2-4:self.player.y//2+4,3]
        life_count = np.count_nonzero(player_region)
        self.player.life -= life_count

        if self.player.life <= 0:
            self.scene = SCENE_GAMEOVER
        
        self.player.update()
        self.gnca.update()
        self.gs.update()
        update_entities(bullets)
        update_entities(enemy_seeds)
        cleanup_entities(bullets)
        cleanup_entities(enemy_seeds)
    
    def update_gameover_scene(self):
        update_entities(bullets)
        update_entities(enemy_seeds)
        cleanup_entities(bullets)
        cleanup_entities(enemy_seeds)

        if pyxel.btnp(pyxel.KEY_RETURN):
            self.scene = SCENE_PLAY
            enemy_seeds.clear()
            bullets.clear()
            gncas.clear()
            self.player = Player(60,100)
            self.gnca = GNCA(position_x = x, position_y = y)
            self.gs = GS(position_x = 0, position_y = 0)
    
    def draw_title_scene(self):
        pyxel.text(40, 40, "NCA GAME", 7)
        pyxel.text(30, 60, "PRESS ENTER KEY", 7)
    
    def draw_gameover_scene(self):
        pyxel.text(40, 40, "GAME OVER", 7)
        pyxel.text(30, 60, "PRESS ENTER KEY", 7)

    def draw_play_scene(self):
        self.player.draw()
        researd = self.gnca.draw()
        grayscott = self.gs.draw()
        draw_nca(researd, grayscott)
        draw_entities(bullets)
        draw_entities(enemy_seeds)
        pyxel.text(3, 3, f"LIFE", 7)
        pyxel.rect(22,3, MAX_LIFE//30, 5, 10)
        pyxel.rect(22,3, self.player.life//30, 5, 11)
        
App()