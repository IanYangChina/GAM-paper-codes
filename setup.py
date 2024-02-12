from setuptools import setup, find_packages


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'manipulation_project' or p.startswith('manipulation_project.')

setup(name='manipulation_project',
      version='0.0.1',
      author='XintongYang',
      author_email='YangX66@cardiff.ac.uk',
      package_data={'manipulation_project': [
          'graspnet_agent/graspnet_checkpoints/*.tar',
          'graspnet_agent/graspnet_checkpoints/*.png',
          'bpbot_agent/cfg/config.yaml',
          'env/assets/objects/meshes/*.stl',
          'env/assets/objects/xml/*.xml',
          'env/assets/robot/kuka/*.xml',
          'env/assets/robot/meshes/kuka/*.obj',
          'env/assets/robot/meshes/kuka/*.stl',
          'env/assets/robot/meshes/kuka/*.mtl',
          'env/assets/robot/gripper/*.xml',
          'env/assets/robot/meshes/robotiq_140_gripper/*.obj',
          'env/assets/robot/meshes/robotiq_140_gripper/*.stl',
          'env/assets/robot/meshes/rethink_gripper/*.stl',
          'env/assets/textures/*.png',
          'env/assets/scene/*.xml',
          'example_configs/*.json'
      ]},
      packages=packages,
      package_dir={'manipulation_project': 'manipulation_project'},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ])
