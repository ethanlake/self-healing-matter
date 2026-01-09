import os
import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

os.environ['FFMPEG_BINARY'] = 'ffmpeg'

# Optional import for Jupyter notebook display
try:
    import moviepy.editor as mvp
    HAS_MOVIEPY_EDITOR = True
except ImportError:
    HAS_MOVIEPY_EDITOR = False


class VideoWriter:
    def __init__(self, filename='tmp.mp4', fps=30.0, autoplay=False, **kw):
        self.writer = None
        self.autoplay = autoplay
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.autoplay:
            self.show()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        if HAS_MOVIEPY_EDITOR:
            try:
                display(mvp.ipython_display(fn, **kw))
            except NameError:
                # display() is not available outside Jupyter
                print(f"Video saved to: {fn}")
        else:
            print(f"Video saved to: {fn}")
