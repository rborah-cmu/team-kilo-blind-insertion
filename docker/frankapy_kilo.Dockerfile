# Course frankapy image + MuJoCo for Team Kilo's IK-based localization and insertion.
#   docker build -t frankapy_docker:kilo - < docker/frankapy_kilo.Dockerfile
# The course image (frankapy_docker:latest) is not modified. --no-deps keeps the image's
# pinned numpy 1.23.5 / scipy 1.8 (frankapy is built against them).
FROM frankapy_docker:latest
RUN pip3 install --no-cache-dir --no-deps \
        mujoco==3.2.3 absl-py==2.3.1 etils==1.3.0 glfw==2.11.0 pyopengl==3.1.10 \
        importlib_resources==6.4.5 zipp==3.20.2 typing_extensions==4.13.2 \
    && python3 -c "import mujoco, numpy, scipy; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)"
