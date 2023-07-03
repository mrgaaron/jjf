# Copyright 2023 Aaron Brenzel.
#
# Licensed under the MIT Licence;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A setuptools based setup module for note-seq."""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "astropy",
    "librosa >= 0.9.2",
    "numpy == 1.24.4",
    "pandas >= 1.5.2",
    "pretty_midi >= 0.2.9",
    "scipy >= 1.11.1",
    "tensorflow >= 2.12.0",
    "urllib3 < 2.0",
]

setup(
    name="jjf",
    version="0.0.1",  # pylint: disable=undefined-variable
    description="Use machine learning to create art and music",
    long_description="",
    author="Aaron Brenzel",
    author_email="aaronbrenzel@gmail.com",
    license="Apache 2",
    keywords="note_seq note sequences",
    packages=find_packages(),
    package_data={"jjf": ["*.pyi", "**/*.pyi"]},
    install_requires=REQUIRED_PACKAGES,
    setup_requires=["pytest-runner", "pytest-pylint"],
    tests_require=[
        "pytest >= 5.2.0",
        "pytest-xdist < 1.30.0",  # 1.30 has problems working with pylint plugin
        "pylint >= 2.4.2",
    ],
)
