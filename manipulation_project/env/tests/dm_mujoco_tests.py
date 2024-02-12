import os, math
import numpy as np
import mujoco
import glfw


def init_window(max_width, max_height):
    glfw.init()
    win = glfw.create_window(width=max_width, height=max_height,
                                       title='Demo', monitor=None,
                                       share=None)
    glfw.make_context_current(win)
    return win


window = init_window(1200, 900)
model_path = "scene/test_scene.xml"
fullpath = os.path.join(os.path.dirname(__file__), '..', 'assets', model_path)
model = mujoco.MjModel.from_xml_path(filename=fullpath, assets=dict())
data = mujoco.MjData(model)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

width, height = glfw.get_framebuffer_size(window)
viewport = mujoco.MjrRect(0, 0, width, height)

scene = mujoco.MjvScene(model, 1000)
camera = mujoco.MjvCamera()
mujoco.mjv_updateScene(
    model, data, mujoco.MjvOption(), mujoco.MjvPerturb(),
    camera, mujoco.mjtCatBit.mjCAT_ALL, scene)

while not glfw.window_should_close(window):
    mujoco.mj_step(model, data)

    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None,
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
