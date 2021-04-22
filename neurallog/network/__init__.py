#  Copyright 2021 Victor Guimarães
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Package to store the network modules.
"""
from typing import Callable


def registry(func: Callable, identifier, func_dict):
    """
    Registries the function or class.

    :param func: the function
    :type func: Callable
    :param identifier: the function identifier
    :type identifier: str
    :param func_dict: the dictionary to registry the function in
    :type func_dict: dict[str, Any]
    :return: the function
    :rtype: Callable
    """
    func_dict[identifier] = func
    return func
