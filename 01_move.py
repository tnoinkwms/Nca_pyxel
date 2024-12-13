import pyxel

class Player:
    def __init__(self, x, y):
        # 表示位置の座標を示す変数
        self.x = x
        self.y = y

class App:
    def __init__(self):
        pyxel.init(160, 120, fps=10, title="move rabbit")
        pyxel.load("my_resource.pyxres")
        self.player = Player(0, 55)
        self.img = [0,0,8,8]
        pyxel.run(self.update, self.draw)
    def update(self):
        if pyxel.btnp(pyxel.KEY_LEFT):
            self.player.x -= 8
            self.img =  [0,0,8,8]
        elif pyxel.btnp(pyxel.KEY_RIGHT):
            self.player.x += 8
            self.img = [8,8,15,15]
        elif pyxel.btnp(pyxel.KEY_UP):
            self.player.y -= 8
            self.img = [48,8,55,15]
        elif pyxel.btnp(pyxel.KEY_DOWN):
            self.player.y += 8
            self.img = [48,0,55,8]
    def draw(self):
        pyxel.cls(0)
        pyxel.blt(self.player.x, self.player.y,0, self.img[0], self.img[1], self.img[2], self.img[3])
        pyxel.text(55, 30, "Hello, Rabbit!", pyxel.frame_count % 16)

App()