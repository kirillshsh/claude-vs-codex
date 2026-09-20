"""Базовый класс сцены. Сцена — чистая функция времени: render(t) -> холст float32 (270x480x3, 0..255).
Никакого состояния между кадрами (кадры рендерятся параллельно и в произвольном порядке)."""
from px import W, H, canvas


class Scene:
    name = 'scene'
    dur = 5.0

    def render(self, t):
        raise NotImplementedError
