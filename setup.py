
import sys
import os
import shutil

if sys.platform.startswith('win'):
    import ctypes
    import comtypes

    VARIANT_BOOL = ctypes.c_bool
    comtypes.VARIANT_BOOL = VARIANT_BOOL
    setattr(sys.modules['comtypes'], 'VARIANT_BOOL', VARIANT_BOOL)

    import pyMSVC
    env = pyMSVC.setup_environment()

    print(env)
    SHELL = 'cmd'
else:
    SHELL = 'bash'


from setuptools import setup
import subprocess


DUMMY_RETURN = b''


def spawn(cmd):
    cmd += '\n'
    cmd = cmd.encode('utf-8')

    if sys.platform.startswith('win'):
        p = subprocess.Popen(
            SHELL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            env=os.environ
        )
    else:
        p = subprocess.Popen(
            SHELL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            env=os.environ
        )

    p.stdin.write(cmd)
    p.stdin.close()

    while p.poll() is None:
        for line in iter(p.stdout.readline, DUMMY_RETURN):
            line = line.strip()
            if line:
                if sys.platform.startswith('win'):
                    sys.stdout.write(line.decode('utf-8') + '\n')
                else:
                    sys.stdout.write(line.decode('utf-8') + '\n')

                sys.stdout.flush()

        for line in iter(p.stderr.readline, DUMMY_RETURN):
            line = line.strip()
            if line:
                sys.stderr.write(line.decode('utf-8') + '\n')
                sys.stderr.flush()

    if not p.stdout.closed:
        p.stdout.close()

    if not p.stderr.closed:
        p.stderr.close()

    sys.stdout.flush()
    sys.stderr.flush()


p_cmd = (
    'cd libs/assimp&&'
    'cmake -G Ninja -DASSIMP_BUILD_TESTS=off -DASSIMP_INSTALL=off -S . -B build&&'
    'cd build&&'
    'ninja'
)

spawn(p_cmd)

data_files = []

for file in os.listdir('libs/assimp/build/bin'):
    src = f'libs/assimp/build/bin/{file}'
    dst = f'wxOpenGL/{file}'
    shutil.copyfile(src=src, dst=dst)
    data_files.append(os.path.abspath(dst))


base_path = os.path.dirname(__file__)

setup(
    name="wxOpenGL",
    version="0.1.0",
    author='Kevin G. Schlosser',
    description="OpenGL framework using the wxPython GUI framework.",
    license="MIT",
    keywords="",
    url="https://github.com/kdschlosser/wxOpenGL",
    packages=['wxOpenGL'],
    include_package_data=True,
    package_data={'wxOpenGL': data_files},
    install_requires=[
        "PyOpenGL>=3.1.10",
        "PyOpenGL-accelerate>=3.1.10",
        "cadquery-ocp==7.8.1.1",
        "numpy==2.2.6",
        "pyfqmr==0.5.0",
        "wxPython==4.2.4",
        "pillow==12.1.0",
        "pyassimp @ file:///" + os.path.join(base_path, 'libs/assimp/port/PyAssimp')
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only"
    ]
)
