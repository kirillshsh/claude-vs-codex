from px import *
from chars import load
from PIL import Image
cl, cx = load()
c = canvas()
vgradient(c, 0, 200, [hexc('#3a7bd5'), hexc('#5b9be6'), hexc('#8cc4f5'), hexc('#bfe3ff')])
rect(c, 0, 200, W, 70, hexc('#58b24f'))
rect(c, 0, 200, W, 1, hexc('#8fdc6a'))
x = 20
for kind in ['idle', 'blink', 'happy', 'angry', 'furious', 'sad', 'surprised', 'dizzy', 'love', 'back']:
    cl.draw_face(c, kind, x + 24, 120, k=2, blush=(kind == 'love'))
    x += 46
x = 20
for kind in ['idle', 'happy', 'angry', 'sad', 'love']:
    cl.draw_face(c, kind, x + 48, 196, k=4, blush=kind in ('love', 'happy'), tears=(kind == 'sad'))
    x += 100
cx.draw(c, 0, 0, 30, 60, 1)
cx.draw(c, 0, 0, 70, 60, 1, face='heart')
cx.draw(c, 0, 0, 110, 60, 1, face='happy')
cx.draw(c, 0, 0, 150, 60, 1, face='back')
cx.draw(c, 0, 0, 190, 60, 1, face='lt3')
cx.draw(c, 5, 1, 230, 60, 1)
cx.draw(c, 0, 0, 300, 110, 2, face='heart')
cx.draw(c, 0, 0, 380, 110, 2, face='back')
cx.draw(c, 8, 1, 450, 110, 2)
b = bubble_sprite('Я ЛУЧШЕ!', tail='left')
blit(c, b, 250, 4)
b = bubble_sprite('НЕТ, Я ЛУЧШЕ!', tail='right')
blit(c, b, 340, 4)
t = text_sprite('CLAWD VS CODEX', (255, 220, 90), scale=2, outline=(40, 20, 10), shadow=(40, 20, 10))
blit(c, t, 10, 232)
t2 = text_sprite('пиксельная история дружбы', (255, 255, 255), font='small', outline=(20, 20, 40))
blit(c, t2, 260, 240)
im = Image.fromarray(np.clip(c, 0, 255).astype(np.uint8)).resize((W * 4, H * 4), Image.NEAREST)
im.save('test_chars.png')
